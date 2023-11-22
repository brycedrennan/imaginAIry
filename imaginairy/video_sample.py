import logging
import math
import os
import random
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from imaginairy import config
from imaginairy.model_manager import get_cached_url_path
from imaginairy.paths import PKG_ROOT
from imaginairy.utils import (
    default,
    get_device,
    instantiate_from_config,
    platform_appropriate_autocast,
)

logger = logging.getLogger(__name__)


def generate_video(
    input_path: str = "other/images/sound-music.jpg",  # Can either be image file or folder with image files
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    model_name: str = "svd_xt",
    fps_id: int = 6,
    output_fps: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: Optional[int] = None,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: Optional[str] = None,
    output_folder: Optional[str] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    device = default(device, get_device)
    seed = default(seed, random.randint(0, 1000000))
    output_fps = default(output_fps, fps_id)

    logger.info(f"Device: {device} seed: {seed}")

    torch.cuda.reset_peak_memory_stats()
    video_model_config = config.video_models.get(model_name, None)
    if video_model_config is None:
        msg = f"Version {model_name} does not exist."
        raise ValueError(msg)

    num_frames = default(num_frames, video_model_config["default_frames"])
    num_steps = default(num_steps, video_model_config["default_steps"])
    output_folder = default(output_folder, "outputs/video/")
    video_config_path = f"{PKG_ROOT}/{video_model_config['config_path']}"

    model, safety_filter = load_model(
        config=video_config_path,
        device=device,
        num_frames=num_frames,
        num_steps=num_steps,
        weights_url=video_model_config["weights_url"],
    )
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any(input_path.endswith(x) for x in ["jpg", "jpeg", "png"]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = (x - x % 64 for x in (w, h))
                image = image.resize((width, height))
                logger.info(
                    f"Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            logger.warning(
                "The image you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            logger.warning(
                "High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            logger.warning("Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            logger.warning("Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad(), platform_appropriate_autocast():
            reload_model(model.conditioner)
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )
            unload_model(model.conditioner)

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(_input, sigma, c):
                _input = _input.half()
                return model.denoiser(
                    model.model, _input, sigma, c, **additional_model_inputs
                )

            reload_model(model.denoiser)
            reload_model(model.model)
            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            unload_model(model.model)
            unload_model(model.denoiser)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            base_count = len(glob(os.path.join(output_folder, "*.mp4")))
            video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MP4V"),
                output_fps,
                (samples.shape[-1], samples.shape[-2]),
            )

            samples = safety_filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            for frame in vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            writer.release()
        if torch.cuda.is_available():
            peak_memory_usage = torch.cuda.max_memory_allocated()
            msg = f"Peak memory usage: {peak_memory_usage / (1024 ** 2)} MB"
            logger.info(msg)
        logger.info(f"Video saved to {video_path}\n")


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch:
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str, device: str, num_frames: int, num_steps: int, weights_url: str
):
    config = OmegaConf.load(config)
    ckpt_path = get_cached_url_path(weights_url)
    config["model"]["params"]["ckpt_path"] = ckpt_path
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )

    model = instantiate_from_config(config.model).to(device).half().eval()

    # safety_filter = DeepFloydDataFiltering(verbose=False, device=device)
    def safety_filter(x):
        return x

    # use less memory
    model.model.half()
    return model, safety_filter


lowvram_mode = True


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        if get_device() == "cuda":
            torch.cuda.empty_cache()


def reload_model(model):
    model.to(get_device())


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    generate_video()

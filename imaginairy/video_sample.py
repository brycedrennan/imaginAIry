"""Functions for generating synthetic videos"""

import logging
import math
import os
import random
import re
import time
from glob import glob
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from imaginairy import config
from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import (
    default,
    get_device,
    instantiate_from_config,
    platform_appropriate_autocast,
)
from imaginairy.utils.model_manager import get_cached_url_path
from imaginairy.utils.paths import PKG_ROOT

logger = logging.getLogger(__name__)


def generate_video(
    input_path: str,  # Can either be image file or folder with image files
    output_folder: str | None = None,
    num_frames: int = 6,
    num_steps: int = 30,
    model_name: str = "svd_xt",
    fps_id: int = 6,
    output_fps: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: Optional[int] = None,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: Optional[str] = None,
    repetitions=1,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    device = default(device, get_device)

    if device == "mps":
        msg = "Apple Silicon MPS (M1, M2, etc) is not currently supported for video generation. Switching to cpu generation."
        logger.warning(msg)
        device = "cpu"

    elif not torch.cuda.is_available():
        msg = (
            "CUDA is not available. This will be verrrry slow or not work at all.\n"
            "If you have a GPU, make sure you have CUDA installed and PyTorch is compiled with CUDA support.\n"
            "Unfortunately, we cannot automatically install the proper version.\n\n"
            "You can install the proper version by following these directions:\n"
            "https://pytorch.org/get-started/locally/"
        )
        logger.warning(msg)

    start_time = time.perf_counter()
    seed = default(seed, random.randint(0, 1000000))
    output_fps = default(output_fps, fps_id)

    video_model_config = config.MODEL_WEIGHT_CONFIG_LOOKUP.get(model_name, None)
    if video_model_config is None:
        msg = f"Version {model_name} does not exist."
        raise ValueError(msg)

    num_frames = default(num_frames, video_model_config.defaults.get("frames", 12))
    num_steps = default(num_steps, video_model_config.defaults.get("steps", 30))
    output_folder_str = default(output_folder, "outputs/video/")
    del output_folder
    video_config_path = f"{PKG_ROOT}/{video_model_config.architecture.config_path}"

    logger.info(
        f"Generating a {num_frames} frame video from {input_path}. Device:{device} seed:{seed}"
    )
    model, safety_filter = load_model(
        config=video_config_path,
        device="cpu",
        num_frames=num_frames,
        num_steps=num_steps,
        weights_url=video_model_config.weights_location,
    )
    torch.manual_seed(seed)

    if input_path.startswith("http"):
        all_img_paths = [input_path]
    else:
        path = Path(input_path)
        if path.is_file():
            if any(input_path.endswith(x) for x in ["jpg", "jpeg", "png"]):
                all_img_paths = [input_path]
            else:
                raise ValueError("Path is not valid image file.")
        elif path.is_dir():
            all_img_paths = sorted(
                [
                    str(f)
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
            )
            if len(all_img_paths) == 0:
                raise ValueError("Folder does not contain any images.")
        else:
            msg = f"Could not find file or folder at {input_path}"
            raise FileNotFoundError(msg)

    expected_size = (1024, 576)
    for _ in range(repetitions):
        for input_path in all_img_paths:
            if input_path.startswith("http"):
                image = LazyLoadingImage(url=input_path).as_pillow()
            else:
                image = LazyLoadingImage(filepath=input_path).as_pillow()
            crop_coords = None
            if image.mode == "RGBA":
                image = image.convert("RGB")
            if image.size != expected_size:
                logger.info(
                    f"Resizing image from {image.size} to {expected_size}. (w, h)"
                )
                image = pillow_fit_image_within(
                    image, max_height=expected_size[1], max_width=expected_size[0]
                )
                logger.debug(f"Image is now of size: {image.size}")
                background = Image.new("RGB", expected_size, "white")
                # Calculate the position to center the original image
                x = (background.width - image.width) // 2
                y = (background.height - image.height) // 2
                background.paste(image, (x, y))
                # crop_coords = (x, y, x + image.width, y + image.height)

                # image = background
            w, h = image.size
            snap_to = 64
            if h % snap_to != 0 or w % snap_to != 0:
                width = w - w % snap_to
                height = h - h % snap_to
                image = image.resize((width, height))
                logger.warning(
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
            if expected_size != (W, H):
                logger.warning(
                    f"The {W, H} image you provided is not {expected_size}.  This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
                )
            if motion_bucket_id > 255:
                logger.warning(
                    "High motion bucket! This may lead to suboptimal performance."
                )

            if fps_id < 5:
                logger.warning(
                    "Small fps value! This may lead to suboptimal performance."
                )

            if fps_id > 30:
                logger.warning(
                    "Large fps value! This may lead to suboptimal performance."
                )

            value_dict: dict[str, Any] = {}
            value_dict["motion_bucket_id"] = motion_bucket_id
            value_dict["fps_id"] = fps_id
            value_dict["cond_aug"] = cond_aug
            value_dict["cond_frames_without_noise"] = image
            value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
            value_dict["cond_aug"] = cond_aug

            with torch.no_grad(), platform_appropriate_autocast():
                reload_model(model.conditioner, device=device)
                if device == "cpu":
                    model.conditioner.to(torch.float32)
                for k in value_dict:
                    if isinstance(value_dict[k], torch.Tensor):
                        value_dict[k] = value_dict[k].to(
                            next(model.conditioner.parameters()).dtype
                        )
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
                    _input = _input.half().to(device)
                    return model.denoiser(
                        model.model, _input, sigma, c, **additional_model_inputs
                    )

                reload_model(model.denoiser, device=device)
                reload_model(model.model, device=device)
                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                reload_model(model.first_stage_model, device=device)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if crop_coords:
                    left, upper, right, lower = crop_coords
                    samples = samples[:, :, upper:lower, left:right]

                os.makedirs(output_folder_str, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder_str, "*.mp4"))) + 1
                source_slug = make_safe_filename(input_path)
                video_filename = f"{base_count:06d}_{model_name}_{seed}_{fps_id}fps_{source_slug}.mp4"
                video_path = os.path.join(output_folder_str, video_filename)
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),  # type: ignore
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
                video_path_h264 = video_path[:-4] + "_h264.mp4"
                os.system(f"ffmpeg -i {video_path} -c:v libx264 {video_path_h264}")

            duration = time.perf_counter() - start_time
            logger.info(
                f"Video of {num_frames} frames generated in {duration:.2f} seconds and saved to {video_path}\n"
            )


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
    oconfig = OmegaConf.load(config)
    ckpt_path = get_cached_url_path(weights_url)
    oconfig["model"]["params"]["ckpt_path"] = ckpt_path  # type: ignore
    if device == "cuda":
        oconfig.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    oconfig.model.params.sampler_config.params.num_steps = num_steps
    oconfig.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )

    model = instantiate_from_config(oconfig.model).to(device).half().eval()

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


def reload_model(model, device=None):
    device = default(device, get_device)
    model.to(device)


def pillow_fit_image_within(
    image: Image.Image, max_height=512, max_width=512, convert="RGB", snap_size=8
):
    image = image.convert(convert)
    w, h = image.size
    resize_ratio = 1

    if w > max_width or h > max_height:
        resize_ratio = min(max_width / w, max_height / h)
    elif w < max_width and h < max_height:
        # it's smaller than our target image, enlarge
        resize_ratio = min(max_width / w, max_height / h)

    if resize_ratio != 1:
        w, h = int(w * resize_ratio), int(h * resize_ratio)
    # resize to integer multiple of snap_size
    w -= w % snap_size
    h -= h % snap_size

    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    return image


def make_safe_filename(input_string):
    stripped_url = re.sub(r"^https?://[^/]+/", "", input_string)

    # Remove directory path if present
    base_name = os.path.basename(stripped_url)

    # Remove file extension
    name_without_extension = os.path.splitext(base_name)[0]

    # Keep only alphanumeric characters and dashes
    safe_name = re.sub(r"[^a-zA-Z0-9\-]", "", name_without_extension)

    return safe_name

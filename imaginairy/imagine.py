import argparse
import os
import random
import re
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast

from imaginairy.models.diffusion.ddim import DDIMSampler
from imaginairy.models.diffusion.plms import PLMSSampler
from imaginairy.utils import get_device, instantiate_from_config

LIB_PATH = os.path.dirname(__file__)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, nargs="?", default=None, help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--plms", action="store_true", help="use plms sampling", default=True
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="how many images",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="data/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    opt = parser.parse_args()
    if opt.plms:
        sampler = "PLMS"
    else:
        sampler = "DDIM"
    prompt = ImaginePrompt(
        opt.prompt,
        seed=opt.seed,
        sampler_type=sampler,
        steps=opt.steps,
        height=opt.H,
        width=opt.W,
        upscale=True,
        fix_faces=True,
    )
    imagine(
        [prompt],
        config=opt.config,
        ckpt=opt.ckpt,
        outdir=opt.outdir,
        fixed_code=opt.fixed_code,
        latent_channels=opt.C,
        precision=opt.precision,
        downsampling_factor=opt.f,
        skip_save=opt.skip_save,
        ddim_eta=opt.ddim_eta,
    )


class WeightedPrompt:
    def __init__(self, text, weight=1):
        self.text = text
        self.weight = weight

    def __str__(self):
        return f"{self.weight}*({self.text})"


class ImaginePrompt:
    def __init__(
        self,
        prompt=None,
        seed=None,
        prompt_strength=7.5,
        sampler_type="PLMS",
        steps=50,
        height=512,
        width=512,
        upscale=True,
        fix_faces=True,
        parts=None,
    ):
        prompt = prompt if prompt is not None else "a scenic landscape"
        if isinstance(prompt, str):
            self.prompts = [WeightedPrompt(prompt, 1)]
        else:
            self.prompts = prompt
        self.prompts.sort(key=lambda p: p.weight, reverse=True)
        self.seed = random.randint(1, 1_000_000_000) if seed is None else seed
        self.prompt_strength = prompt_strength
        self.sampler_type = sampler_type
        self.steps = steps
        self.height = height
        self.width = width
        self.upscale = upscale
        self.fix_faces = fix_faces
        self.parts = parts or {}

    @property
    def prompt_text(self):
        if len(self.prompts) == 1:
            return self.prompts[0].text
        return "|".join(str(p) for p in self.prompts)


def imagine(
    prompts,
    config="data/stable-diffusion-v1.yaml",
    ckpt="data/stable-diffusion-v1-4.ckpt",
    outdir="outputs/txt2img-samples",
    fixed_code=None,
    latent_channels=4,
    downsampling_factor=8,
    precision="autocast",
    skip_save=False,
    ddim_eta=0.0,
):
    config = OmegaConf.load(f"{LIB_PATH}/{config}")
    model = load_model_from_config(config, f"{LIB_PATH}/{ckpt}")

    model = model.to(get_device())

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    sample_path = os.path.join(outpath)
    big_path = os.path.join(sample_path, "esrgan")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(big_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    precision_scope = autocast if precision == "autocast" else nullcontext
    with (torch.no_grad(), precision_scope("cuda")):
        for prompt in prompts:
            seed_everything(prompt.seed)
            uc = None
            if prompt.prompt_strength != 1.0:
                uc = model.get_learned_conditioning(1 * [""])
            total_weight = sum(wp.weight for wp in prompt.prompts)
            c = sum(
                [
                    model.get_learned_conditioning(wp.text) * (wp.weight / total_weight)
                    for wp in prompt.prompts
                ]
            )
            # c = model.get_learned_conditioning(prompt.prompt_text)

            shape = [
                latent_channels,
                prompt.height // downsampling_factor,
                prompt.width // downsampling_factor,
            ]

            def img_callback(samples, i):
                return
                samples = model.decode_first_stage(samples)
                samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                for pred_x0 in samples:
                    pred_x0 = 255.0 * rearrange(pred_x0.cpu().numpy(), "c h w -> h w c")
                    filename = f"{base_count:08}_S{seed}_step{i:04}.jpg"
                    Image.fromarray(pred_x0.astype(np.uint8)).save(
                        os.path.join(sample_path, filename)
                    )

            start_code = None
            if fixed_code:
                start_code = torch.randn(
                    [1, latent_channels, prompt.height, prompt.width],
                    device=get_device(),
                )
            sampler = get_sampler(prompt.sampler_type, model)
            samples_ddim, _ = sampler.sample(
                S=prompt.steps,
                conditioning=c,
                batch_size=1,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=prompt.prompt_strength,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=start_code,
                img_callback=img_callback,
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if not skip_save:
                for x_sample in x_samples_ddim:
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    basefilename = f"{base_count:06}_{prompt.seed}_{prompt.sampler_type}{prompt.steps}_PS{prompt.prompt_strength}_{prompt_normalized(prompt.prompt_text)}"
                    filepath = os.path.join(sample_path, f"{basefilename}.jpg")
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    if prompt.fix_faces:
                        img = fix_faces_GFPGAN(img)
                    img.save(filepath)
                    if prompt.upscale:
                        enlarge_realesrgan2x(
                            filepath,
                            os.path.join(big_path, basefilename) + ".jpg",
                        )
                    base_count += 1
                    return f"{basefilename}.jpg"


def prompt_normalized(prompt):
    return re.sub(r"[^a-zA-Z0-9.,]+", "_", prompt)[:130]


DOWNLOADED_FILES_PATH = f"{LIB_PATH}/../downloads/"
ESRGAN_PATH = DOWNLOADED_FILES_PATH + "realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan"


def enlarge_realesrgan2x(src, dst):
    process = subprocess.Popen(
        [ESRGAN_PATH, "-i", src, "-o", dst, "-n", "realesrgan-x4plus"]
    )
    process.wait()


def get_sampler(sampler_type, model):
    sampler_type = sampler_type.upper()
    if sampler_type == "PLMS":
        return PLMSSampler(model)
    elif sampler_type == "DDIM":
        return DDIMSampler(model)


def gfpgan_model():
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=DOWNLOADED_FILES_PATH
        + "GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=torch.device(get_device()),
    )


def fix_faces_GFPGAN(image):
    image = image.convert("RGB")
    cropped_faces, restored_faces, restored_img = gfpgan_model().enhance(
        np.array(image, dtype=np.uint8),
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    res = Image.fromarray(restored_img)

    return res


if __name__ == "__main__":
    main()

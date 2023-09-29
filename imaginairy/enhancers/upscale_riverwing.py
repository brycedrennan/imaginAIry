from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from imaginairy.log_utils import log_latent
from imaginairy.model_manager import hf_hub_download
from imaginairy.utils import get_device, platform_appropriate_autocast
from imaginairy.vendored import k_diffusion as K
from imaginairy.vendored.k_diffusion import layers
from imaginairy.vendored.k_diffusion.models.image_v1 import ImageDenoiserModelV1
from imaginairy.vendored.k_diffusion.utils import append_dims


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, inp, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        sigma_data = self.sigma_data
        # 'MPS does not support power op with int64 input'
        if isinstance(low_res_sigma, torch.Tensor):
            low_res_sigma = low_res_sigma.to(torch.float32)
        if isinstance(sigma_data, torch.Tensor):
            sigma_data = sigma_data.to(torch.float32)
        c_in = 1 / (low_res_sigma**2 + sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode="nearest") * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            inp,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


@lru_cache(maxsize=1)
def get_upscaler_model(
    model_path,
    pooler_dim=768,
    train=False,
    device=get_device(),
):
    config = {
        "type": "image_v1",
        "input_channels": 4,
        "input_size": [48, 48],
        "patch_size": 1,
        "mapping_out": 768,
        "mapping_cond_dim": 896,
        "unet_cond_dim": 4,
        "depths": [4, 4, 4, 4],
        "channels": [384, 384, 768, 768],
        "self_attn_depths": [False, False, False, True],
        "cross_attn_depths": [False, True, True, True],
        "cross_cond_dim": 768,
        "has_variance": True,
        "dropout_rate": 0.0,
        "augment_prob": 0.0,
        "augment_wrapper": False,
        "sigma_data": 1.0,
        "sigma_min": 1e-2,
        "sigma_max": 20,
        "sigma_sample_density": {"type": "lognormal", "mean": -0.5, "std": 1.2},
        "skip_stages": 0,
    }

    model = ImageDenoiserModelV1(
        config["input_channels"],
        config["mapping_out"],
        config["depths"],
        config["channels"],
        config["self_attn_depths"],
        config["cross_attn_depths"],
        patch_size=config["patch_size"],
        dropout_rate=config["dropout_rate"],
        mapping_cond_dim=config["mapping_cond_dim"]
        + (9 if config["augment_wrapper"] else 0),
        unet_cond_dim=config["unet_cond_dim"],
        cross_cond_dim=config["cross_cond_dim"],
        skip_stages=config["skip_stages"],
        has_variance=config["has_variance"],
    )

    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config["sigma_data"],
        embed_dim=config["mapping_cond_dim"] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_ema"])
    model = layers.DenoiserWithVariance(model, sigma_data=config["sigma_data"])
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale, device):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale
        self.device = device

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            elif self.cond_scale == 1.0:
                c_in = c
            return self.inner_model(
                x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in
            )

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(
            x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in
        ).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok_out["input_ids"][indexer]
        attention_mask = 1 - tok_out["attention_mask"][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)."""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging

        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(
            input_ids=input_ids.to(self.device), output_hidden_states=True
        )
        return (
            clip_out.hidden_states[-1],
            cross_cond_padding.to(self.device),
            clip_out.pooler_output,
        )


@lru_cache
def clip_up_models():
    with platform_appropriate_autocast():
        tok_up = CLIPTokenizerTransform()
        text_encoder_up = CLIPEmbedder(device=get_device())
    return text_encoder_up, tok_up


@torch.no_grad()
def condition_up(prompts):
    text_encoder_up, tok_up = clip_up_models()
    return text_encoder_up(tok_up(prompts))


@torch.no_grad()
def upscale_latent(
    low_res_latent,
    upscale_prompt="",
    seed=0,
    steps=15,
    guidance_scale=1.0,
    batch_size=1,
    num_samples=1,
    # Amount of noise to add per step (0.0=deterministic). Used in all samplers except `k_euler`.
    eta=0.1,
    device=get_device(),
):
    # Add noise to the latent vectors before upscaling. This theoretically can make the model work better on out-of-distribution inputs, but mostly just seems to make it match the input less, so it's turned off by default.
    noise_aug_level = 0  # @param {type: 'slider', min: 0.0, max: 0.6, step:0.025}
    noise_aug_type = "fake"  # @param ["gaussian", "fake"]

    # @markdown Sampler settings. `k_dpm_adaptive` uses an adaptive solver with error tolerance `tol_scale`, all other use a fixed number of steps.
    sampler = "k_dpm_2_ancestral"  # @param ["k_euler", "k_euler_ancestral", "k_dpm_2_ancestral", "k_dpm_fast", "k_dpm_adaptive"]

    tol_scale = 0.25  # @param {type: 'number'}

    # seed_everything(seed)

    # uc = condition_up(batch_size * ["blurry, low resolution, 720p, grainy"])
    uc = condition_up(batch_size * [""])
    c = condition_up(batch_size * [upscale_prompt])

    [_, C, H, W] = low_res_latent.shape

    # Noise levels from stable diffusion.
    sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512
    model_up = get_upscaler_model(
        model_path=hf_hub_download(
            "pcuenq/k-upscaler", "laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
        ),
        device=device,
    )
    model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale, device=device)
    low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
    x_shape = [batch_size, C, 2 * H, 2 * W]

    def do_sample(noise, extra_args):
        # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
        sigmas = (
            torch.linspace(np.log(sigma_max), np.log(sigma_min), steps + 1)
            .exp()
            .to(device)
        )
        if sampler == "k_euler":
            return K.sampling.sample_euler(
                model_wrap, noise * sigma_max, sigmas, extra_args=extra_args
            )
        if sampler == "k_euler_ancestral":
            return K.sampling.sample_euler_ancestral(
                model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta
            )
        if sampler == "k_dpm_2_ancestral":
            return K.sampling.sample_dpm_2_ancestral(
                model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta
            )
        if sampler == "k_dpm_fast":
            return K.sampling.sample_dpm_fast(
                model_wrap,
                noise * sigma_max,
                sigma_min,
                sigma_max,
                steps,
                extra_args=extra_args,
                eta=eta,
            )
        if sampler == "k_dpm_adaptive":
            sampler_opts = {
                "s_noise": 1.0,
                "rtol": tol_scale * 0.05,
                "atol": tol_scale / 127.5,
                "pcoeff": 0.2,
                "icoeff": 0.4,
                "dcoeff": 0,
            }
            return K.sampling.sample_dpm_adaptive(
                model_wrap,
                noise * sigma_max,
                sigma_min,
                sigma_max,
                extra_args=extra_args,
                eta=eta,
                **sampler_opts,
            )
        msg = f"Unknown sampler {sampler}"
        raise ValueError(msg)

    for _ in range((num_samples - 1) // batch_size + 1):
        if noise_aug_type == "gaussian":
            latent_noised = low_res_latent + noise_aug_level * torch.randn_like(
                low_res_latent
            )
        elif noise_aug_type == "fake":
            latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
        extra_args = {
            "low_res": latent_noised,
            "low_res_sigma": low_res_sigma,
            "c": c,
        }
        noise = torch.randn(x_shape, device=device)
        up_latents = do_sample(noise, extra_args)
        log_latent(low_res_latent, "low_res_latent")
        return up_latents

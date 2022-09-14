import logging

import numpy as np
import torch
from einops import rearrange
from PIL import Image

_CURRENT_LOGGING_CONTEXT = None

logger = logging.getLogger(__name__)


def log_latent(latents, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    if torch.isnan(latents).any() or torch.isinf(latents).any():
        logger.error(
            "Inf/NaN values showing in transformer."
            + repr(latents)[:50]
            + " "
            + description[:50]
        )
    _CURRENT_LOGGING_CONTEXT.log_latents(latents, description)


class LatentLoggingContext:
    def __init__(self, prompt, model, img_callback=None):
        self.prompt = prompt
        self.model = model
        self.step_count = 0
        self.img_callback = img_callback

    def __enter__(self):
        global _CURRENT_LOGGING_CONTEXT
        _CURRENT_LOGGING_CONTEXT = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_LOGGING_CONTEXT
        _CURRENT_LOGGING_CONTEXT = None

    def log_latents(self, samples, description):
        if not self.img_callback:
            return
        if samples.shape[1] != 4:
            # logger.info(f"Didn't save tensor of shape {samples.shape} for {description}")
            return
        self.step_count += 1
        description = f"{description} - {samples.shape}"
        samples = self.model.decode_first_stage(samples)
        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
        for pred_x0 in samples:
            pred_x0 = 255.0 * rearrange(pred_x0.cpu().numpy(), "c h w -> h w c")
            img = Image.fromarray(pred_x0.astype(np.uint8))
            self.img_callback(img, description, self.step_count, self.prompt)

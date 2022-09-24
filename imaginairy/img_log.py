import logging
import re

import torch
from torchvision.transforms import ToPILImage

from imaginairy.img_utils import model_latents_to_pillow_imgs

_CURRENT_LOGGING_CONTEXT = None

logger = logging.getLogger(__name__)


def log_conditioning(conditioning, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return

    _CURRENT_LOGGING_CONTEXT.log_conditioning(conditioning, description)


def log_latent(latents, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return

    _CURRENT_LOGGING_CONTEXT.log_latents(latents, description)


def log_img(img, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.log_img(img, description)


class ImageLoggingContext:
    def __init__(self, prompt, model, img_callback=None, img_outdir=None):
        self.prompt = prompt
        self.model = model
        self.step_count = 0
        self.img_callback = img_callback
        self.img_outdir = img_outdir

    def __enter__(self):
        global _CURRENT_LOGGING_CONTEXT  # noqa
        _CURRENT_LOGGING_CONTEXT = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_LOGGING_CONTEXT  # noqa
        _CURRENT_LOGGING_CONTEXT = None

    def log_conditioning(self, conditioning, description):
        if not self.img_callback:
            return
        img = conditioning_to_img(conditioning)

        self.img_callback(img, description, self.step_count, self.prompt)

    def log_latents(self, latents, description):
        if not self.img_callback:
            return
        if latents.shape[1] != 4:
            # logger.info(f"Didn't save tensor of shape {samples.shape} for {description}")
            return
        self.step_count += 1
        description = f"{description} - {latents.shape}"
        for img in model_latents_to_pillow_imgs(latents):
            self.img_callback(img, description, self.step_count, self.prompt)

    def log_img(self, img, description):
        if not self.img_callback:
            return
        self.step_count += 1
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img.squeeze().cpu().detach())
        img = img.copy()
        self.img_callback(img, description, self.step_count, self.prompt)

    # def img_callback(self, img, description, step_count, prompt):
    #     steps_path = os.path.join(self.img_outdir, "steps", f"{self.file_num:08}_S{prompt.seed}")
    #     os.makedirs(steps_path, exist_ok=True)
    #     filename = f"{self.file_num:08}_S{prompt.seed}_step{step_count:04}_{filesafe_text(description)[:40]}.jpg"
    #     destination = os.path.join(steps_path, filename)
    #     draw = ImageDraw.Draw(img)
    #     draw.text((10, 10), str(description))
    #     img.save(destination)


def filesafe_text(t):
    return re.sub(r"[^a-zA-Z0-9.,\[\]() -]+", "_", t)[:130]


def conditioning_to_img(conditioning):
    return ToPILImage()(conditioning)

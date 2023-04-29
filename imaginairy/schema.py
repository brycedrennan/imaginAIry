import hashlib
import json
import logging
import os.path
import random
from datetime import datetime, timezone

from imaginairy import config

logger = logging.getLogger(__name__)


class InvalidUrlError(ValueError):
    pass


class LazyLoadingImage:
    def __init__(self, *, filepath=None, url=None):
        if not filepath and not url:
            raise ValueError("You must specify a url or filepath")
        if filepath and url:
            raise ValueError("You cannot specify a url and filepath")

        # validate file exists
        if filepath and not os.path.exists(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")

        # validate url is valid url
        if url:
            from urllib3.exceptions import LocationParseError
            from urllib3.util import parse_url

            try:
                parsed_url = parse_url(url)
            except LocationParseError:
                raise InvalidUrlError(f"Invalid url: {url}")  # noqa
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
                raise InvalidUrlError(f"Invalid url: {url}")

        self._lazy_filepath = filepath
        self._lazy_url = url
        self._img = None

    def __getattr__(self, key):
        if key == "_img":
            #  http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
            raise AttributeError()
        if self._img:
            return getattr(self._img, key)
        from PIL import Image, ImageOps

        if self._lazy_filepath:
            self._img = Image.open(self._lazy_filepath)
            logger.debug(
                f"Loaded input 🖼  of size {self._img.size} from {self._lazy_filepath}"
            )
        elif self._lazy_url:
            import requests

            self._img = Image.open(
                requests.get(self._lazy_url, stream=True, timeout=60).raw
            )
            logger.debug(
                f"Loaded input 🖼  of size {self._img.size} from {self._lazy_url}"
            )
        # fix orientation
        self._img = ImageOps.exif_transpose(self._img)

        return getattr(self._img, key)

    def __str__(self):
        return self._lazy_filepath or self._lazy_url


class WeightedPrompt:
    def __init__(self, text, weight=1):
        self.text = text
        self.weight = weight

    def __str__(self):
        return f"{self.weight}*({self.text})"


class ImaginePrompt:
    class MaskMode:
        KEEP = "keep"
        REPLACE = "replace"

    DEFAULT_FACE_FIDELITY = 0.2

    def __init__(
        self,
        prompt=None,
        negative_prompt=None,
        prompt_strength=7.5,
        init_image=None,  # Pillow Image, LazyLoadingImage, or filepath str
        init_image_strength=None,
        control_image=None,
        control_image_raw=None,
        control_mode=None,
        mask_prompt=None,
        mask_image=None,
        mask_mode=MaskMode.REPLACE,
        mask_modify_original=True,
        outpaint=None,
        seed=None,
        steps=None,
        height=None,
        width=None,
        upscale=False,
        fix_faces=False,
        fix_faces_fidelity=DEFAULT_FACE_FIDELITY,
        sampler_type=config.DEFAULT_SAMPLER,
        conditioning=None,
        tile_mode="",
        allow_compose_phase=True,
        model=config.DEFAULT_MODEL,
        model_config_path=None,
        is_intermediate=False,
        collect_progress_latents=False,
        caption_text="",
    ):
        self.prompts = prompt
        self.negative_prompt = negative_prompt
        self.prompt_strength = prompt_strength
        self.init_image = init_image
        self.init_image_strength = init_image_strength
        self.control_image = control_image
        self.control_image_raw = control_image_raw
        self.control_mode = control_mode
        self._orig_seed = seed
        self.seed = seed
        self.steps = steps
        self.height = height
        self.width = width
        self.upscale = upscale
        self.fix_faces = fix_faces
        self.fix_faces_fidelity = fix_faces_fidelity
        self.sampler_type = sampler_type
        self.conditioning = conditioning
        self.mask_prompt = mask_prompt
        self.mask_image = mask_image
        self.mask_mode = mask_mode
        self.mask_modify_original = mask_modify_original
        self.outpaint = outpaint
        self.tile_mode = tile_mode
        self.allow_compose_phase = allow_compose_phase
        self.model = model
        self.model_config_path = model_config_path
        self.caption_text = caption_text

        # we don't want to save intermediate images
        self.is_intermediate = is_intermediate
        self.collect_progress_latents = collect_progress_latents

        self.validate()

    def validate(self):
        from imaginairy.samplers import SAMPLER_LOOKUP, SamplerName

        self.prompts = self.process_prompt_input(self.prompts)

        if self.tile_mode is True:
            self.tile_mode = "xy"
        elif self.tile_mode is False:
            self.tile_mode = ""
        else:
            self.tile_mode = self.tile_mode.lower()
            assert self.tile_mode in ("", "x", "y", "xy")

        if isinstance(self.control_image, str):
            if not self.control_image.startswith("*prev."):
                self.control_image = LazyLoadingImage(filepath=self.control_image)

        if isinstance(self.control_image_raw, str):
            if not self.control_image_raw.startswith("*prev."):
                self.control_image_raw = LazyLoadingImage(
                    filepath=self.control_image_raw
                )

        if isinstance(self.init_image, str):
            if not self.init_image.startswith("*prev."):
                self.init_image = LazyLoadingImage(filepath=self.init_image)

        if isinstance(self.mask_image, str):
            if not self.mask_image.startswith("*prev."):
                self.mask_image = LazyLoadingImage(filepath=self.mask_image)

        if self.control_image is not None and self.control_image_raw is not None:
            raise ValueError(
                "You can only set one of `control_image` and `control_image_raw`"
            )

        if self.control_image is not None and self.init_image is None:
            self.init_image = self.control_image

        if (
            self.control_mode
            and self.control_image is None
            and self.init_image is not None
        ):
            self.control_image = self.init_image

        if self.control_mode and not (self.control_image or self.control_image_raw):
            raise ValueError("You must set `control_image` when using `control_mode`")

        if self.mask_image is not None and self.mask_prompt is not None:
            raise ValueError("You can only set one of `mask_image` and `mask_prompt`")

        if self.model is None:
            self.model = config.DEFAULT_MODEL

        if self.init_image_strength is None:
            if self.control_mode is not None:
                self.init_image_strength = 0.0
            elif self.outpaint or self.mask_image or self.mask_prompt:
                self.init_image_strength = 0.0
            else:
                self.init_image_strength = 0.2

        self.seed = random.randint(1, 1_000_000_000) if self.seed is None else self.seed

        self.sampler_type = self.sampler_type.lower()

        self.fix_faces_fidelity = (
            self.fix_faces_fidelity
            if self.fix_faces_fidelity
            else self.DEFAULT_FACE_FIDELITY
        )

        if self.height is None or self.width is None or self.steps is None:
            from imaginairy.model_manager import get_model_default_image_size

            SamplerCls = SAMPLER_LOOKUP[self.sampler_type]
            self.steps = self.steps or SamplerCls.default_steps
            self.width = self.width or get_model_default_image_size(self.model)
            self.height = self.height or get_model_default_image_size(self.model)
        self.steps = int(self.steps)
        if self.negative_prompt is None:
            model_config = config.MODEL_CONFIG_SHORTCUTS.get(self.model, None)
            if model_config:
                self.negative_prompt = model_config.default_negative_prompt
            else:
                self.negative_prompt = config.DEFAULT_NEGATIVE_PROMPT

        self.negative_prompt = self.process_prompt_input(self.negative_prompt)

        if self.model == "SD-2.0-v" and self.sampler_type == SamplerName.PLMS:
            raise ValueError("PLMS sampler is not supported for SD-2.0-v model.")

        if self.model == "edit" and self.sampler_type in (
            SamplerName.PLMS,
            SamplerName.DDIM,
        ):
            raise ValueError(
                "PLMS and DDIM samplers are not supported for pix2pix edit model."
            )

    @property
    def prompt_text(self):
        if len(self.prompts) == 1:
            return self.prompts[0].text
        return "|".join(str(p) for p in self.prompts)

    @property
    def negative_prompt_text(self):
        if len(self.negative_prompt) == 1:
            return self.negative_prompt[0].text
        return "|".join(str(p) for p in self.negative_prompt)

    def prompt_description(self):
        return (
            f'"{self.prompt_text}" {self.width}x{self.height}px '
            f'negative-prompt:"{self.negative_prompt_text}" '
            f"seed:{self.seed} prompt-strength:{self.prompt_strength} steps:{self.steps} sampler-type:{self.sampler_type} init-image-strength:{self.init_image_strength} model:{self.model}"
        )

    def as_dict(self):
        prompts = [(p.weight, p.text) for p in self.prompts]
        negative_prompts = [(p.weight, p.text) for p in self.negative_prompt]
        return {
            "software": "imaginAIry",
            "model": self.model,
            "prompts": prompts,
            "prompt_strength": self.prompt_strength,
            "negative_prompt": negative_prompts,
            "init_image": str(self.init_image),
            "init_image_strength": self.init_image_strength,
            # "seed": self.seed,
            "steps": self.steps,
            "height": self.height,
            "width": self.width,
            "upscale": self.upscale,
            "fix_faces": self.fix_faces,
            "sampler_type": self.sampler_type,
        }

    def process_prompt_input(self, prompt_input):
        prompt_input = prompt_input if prompt_input is not None else ""

        if isinstance(prompt_input, str):
            prompt_input = [WeightedPrompt(prompt_input, 1)]

        prompt_input.sort(key=lambda p: p.weight, reverse=True)
        return prompt_input


class ExifCodes:
    """https://www.awaresystems.be/imaging/tiff/tifftags/baseline.html."""

    ImageDescription = 0x010E
    Software = 0x0131
    DateTime = 0x0132
    HostComputer = 0x013C
    UserComment = 0x9286


class ImagineResult:
    def __init__(
        self,
        img,
        prompt: ImaginePrompt,
        is_nsfw,
        safety_score,
        upscaled_img=None,
        modified_original=None,
        mask_binary=None,
        mask_grayscale=None,
        result_images=None,
        timings=None,
        progress_latents=None,
    ):
        import torch

        from imaginairy.img_utils import (
            model_latent_to_pillow_img,
            torch_img_to_pillow_img,
        )
        from imaginairy.utils import get_device, get_hardware_description

        self.prompt = prompt

        self.images = {"generated": img}

        if upscaled_img:
            self.images["upscaled"] = upscaled_img

        if modified_original:
            self.images["modified_original"] = modified_original

        if mask_binary:
            self.images["mask_binary"] = mask_binary

        if mask_grayscale:
            self.images["mask_grayscale"] = mask_grayscale

        for img_type, r_img in result_images.items():
            if isinstance(r_img, torch.Tensor):
                if r_img.shape[1] == 4:
                    r_img = model_latent_to_pillow_img(r_img)
                else:
                    r_img = torch_img_to_pillow_img(r_img)
            self.images[img_type] = r_img

        self.timings = timings
        self.progress_latents = progress_latents

        # for backward compat
        self.img = img
        self.upscaled_img = upscaled_img

        self.is_nsfw = is_nsfw
        self.safety_score = safety_score
        self.created_at = datetime.utcnow().replace(tzinfo=timezone.utc)
        self.torch_backend = get_device()
        self.hardware_name = get_hardware_description(get_device())

    def md5(self):
        return hashlib.md5(self.img.tobytes()).hexdigest()

    def metadata_dict(self):
        return {
            "prompt": self.prompt.as_dict(),
        }

    def timings_str(self):
        if not self.timings:
            return ""
        return " ".join(f"{k}:{v:.2f}s" for k, v in self.timings.items())

    def _exif(self):
        from PIL import Image

        exif = Image.Exif()
        exif[ExifCodes.ImageDescription] = self.prompt.prompt_description()
        exif[ExifCodes.UserComment] = json.dumps(self.metadata_dict())
        # help future web scrapes not ingest AI generated art
        sd_version = self.prompt.model
        if len(sd_version) > 20:
            sd_version = "custom weights"
        exif[ExifCodes.Software] = f"Imaginairy / Stable Diffusion {sd_version}"
        exif[ExifCodes.DateTime] = self.created_at.isoformat(sep=" ")[:19]
        exif[ExifCodes.HostComputer] = f"{self.torch_backend}:{self.hardware_name}"
        return exif

    def save(self, save_path, image_type="generated"):
        img = self.images.get(image_type, None)
        if img is None:
            raise ValueError(
                f"Image of type {image_type} not stored. Options are: {self.images.keys()}"
            )

        img.convert("RGB").save(save_path, exif=self._exif())


class SafetyMode:
    STRICT = "strict"
    RELAXED = "relaxed"

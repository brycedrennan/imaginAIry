import hashlib
import json
import logging
import os.path
import random
from datetime import datetime, timezone
from functools import lru_cache

import requests
from PIL import Image, ImageOps
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

from imaginairy import config
from imaginairy.model_manager import get_model_default_image_size
from imaginairy.samplers import SAMPLER_LOOKUP, SamplerName
from imaginairy.utils import get_device, get_hardware_description

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

        if self._lazy_filepath:
            self._img = Image.open(self._lazy_filepath)
            logger.info(
                f"Loaded input 🖼  of size {self._img.size} from {self._lazy_filepath}"
            )
        elif self._lazy_url:
            self._img = Image.open(
                requests.get(self._lazy_url, stream=True, timeout=60).raw
            )
            logger.info(
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
        negative_prompt=config.DEFAULT_NEGATIVE_PROMPT,
        prompt_strength=7.5,
        init_image=None,  # Pillow Image, LazyLoadingImage, or filepath str
        init_image_strength=0.3,
        mask_prompt=None,
        mask_image=None,
        mask_mode=MaskMode.REPLACE,
        mask_modify_original=True,
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
        model=config.DEFAULT_MODEL,
    ):

        self.prompts = self.process_prompt_input(prompt)
        self.negative_prompt = self.process_prompt_input(negative_prompt)
        self.prompt_strength = prompt_strength
        if tile_mode is True:
            tile_mode = "xy"
        elif tile_mode is False:
            tile_mode = ""
        else:
            tile_mode = tile_mode.lower()
            assert tile_mode in ("", "x", "y", "xy")

        if isinstance(init_image, str):
            init_image = LazyLoadingImage(filepath=init_image)

        if isinstance(mask_image, str):
            mask_image = LazyLoadingImage(filepath=mask_image)

        if mask_image is not None and mask_prompt is not None:
            raise ValueError("You can only set one of `mask_image` and `mask_prompt`")
        if model is None:
            model = config.DEFAULT_MODEL

        self.init_image = init_image
        self.init_image_strength = init_image_strength
        self.seed = random.randint(1, 1_000_000_000) if seed is None else seed
        self.steps = steps
        self.height = height
        self.width = width
        self.upscale = upscale
        self.fix_faces = fix_faces
        self.fix_faces_fidelity = (
            fix_faces_fidelity if fix_faces_fidelity else self.DEFAULT_FACE_FIDELITY
        )
        self.sampler_type = sampler_type.lower()
        self.conditioning = conditioning
        self.mask_prompt = mask_prompt
        self.mask_image = mask_image
        self.mask_mode = mask_mode
        self.mask_modify_original = mask_modify_original
        self.tile_mode = tile_mode
        self.model = model

        if self.height is None or self.width is None or self.steps is None:
            SamplerCls = SAMPLER_LOOKUP[self.sampler_type]
            self.steps = self.steps or SamplerCls.default_steps
            self.width = self.width or get_model_default_image_size(self.model)
            self.height = self.height or get_model_default_image_size(self.model)

        if self.model == "SD-2.0-v" and self.sampler_type == SamplerName.PLMS:
            raise ValueError("PLMS sampler is not supported for SD-2.0-v model.")

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
            f"seed:{self.seed} prompt-strength:{self.prompt_strength} steps:{self.steps} sampler-type:{self.sampler_type}"
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
            "seed": self.seed,
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
    """https://www.awaresystems.be/imaging/tiff/tifftags/baseline.html"""

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
        depth_image=None,
        timings=None,
    ):
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

        if depth_image is not None:
            self.images["depth_image"] = depth_image

        self.timings = timings

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


@lru_cache(maxsize=2)
def _get_briefly_cached_url(url):
    return requests.get(url, timeout=60)

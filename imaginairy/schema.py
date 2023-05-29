# pylint: disable=E0213
import base64
import hashlib
import io
import json
import logging
import os.path
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from pydantic import BaseModel, Field, validator

from imaginairy import config

if TYPE_CHECKING:
    from PIL import Image
else:
    Image = Any


logger = logging.getLogger(__name__)


class InvalidUrlError(ValueError):
    pass


class LazyLoadingImage:
    """Image file encoded as base64 string."""

    def __init__(self, *, filepath=None, url=None, img=None):
        if not filepath and not url and not img:
            raise ValueError("You must specify a url or filepath or img")
        if sum([bool(filepath), bool(url), bool(img)]) > 1:
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
        self._img = img

    def __getattr__(self, key):
        if key == "_img":
            #  http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
            raise AttributeError()
        self._load_img()
        return getattr(self._img, key)

    def _load_img(self):
        if self._img is None:
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
            else:
                raise ValueError("You must specify a url or filepath")
            # fix orientation
            self._img = ImageOps.exif_transpose(self._img)

    @classmethod
    def __modify_schema__(cls, field_schema, field):
        field_schema["title"] = field.name.replace("_", " ").title()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        from PIL import Image

        if isinstance(v, cls):
            return v
        if isinstance(v, Image.Image):
            return cls(img=v)
        if isinstance(v, str):
            return cls(img=cls.load_image_from_base64(v))
        raise ValueError(
            "Image value must be either a PIL.Image.Image or a Base64 string"
        )

    @staticmethod
    def save_image_as_base64(image: "Image.Image") -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode()

    @staticmethod
    def load_image_from_base64(image_str: str) -> "Image.Image":
        from PIL import Image

        img_bytes = base64.b64decode(image_str)
        return Image.open(io.BytesIO(img_bytes))

    def __str__(self):
        self._load_img()
        return self.save_image_as_base64(self._img)  # type: ignore

    def __repr__(self):
        """human readable representation.

        shows filepath or url if available.
        """
        return f"<LazyLoadingImage filepath={self._lazy_filepath} url={self._lazy_url}>"


class ControlNetInput(BaseModel):
    mode: str
    image: Optional[LazyLoadingImage] = None
    image_raw: Optional[LazyLoadingImage] = None
    strength: int = Field(1, ge=0)

    @validator("image_raw")
    def image_raw_validate(cls, v, values):
        if values.get("image") is not None and v is not None:
            raise ValueError("You cannot specify both image and image_raw")

        # if v is None and values.get("image") is None:
        #     raise ValueError("You must specify either image or image_raw")

        return v


class WeightedPrompt(BaseModel):
    text: str
    weight: int = Field(1, ge=0)

    def __repr__(self):
        return f"{self.weight}*({self.text})"


class ImaginePrompt(BaseModel):
    prompt: Optional[List[WeightedPrompt]]
    negative_prompt: Optional[List[WeightedPrompt]]
    prompt_strength: Optional[float] = 7.5
    init_image: Optional[LazyLoadingImage] = Field(
        None, description="base64 encoded image"
    )
    init_image_strength: Optional[float] = Field(ge=0, le=1)
    control_inputs: Optional[List[ControlNetInput]]
    mask_prompt: Optional[str] = Field(
        description="text description of the things to be masked"
    )
    mask_image: Optional[LazyLoadingImage]
    mask_mode: Optional[Literal["keep", "replace"]] = "replace"
    mask_modify_original: bool = True
    outpaint: Optional[str]
    model: str = config.DEFAULT_MODEL
    model_config_path: Optional[str]
    sampler_type: str = config.DEFAULT_SAMPLER
    seed: Optional[int]
    steps: Optional[int]
    height: Optional[int] = Field(None, ge=1)
    width: Optional[int] = Field(None, ge=1)
    upscale: bool = False
    fix_faces: bool = False
    fix_faces_fidelity: Optional[float] = Field(0.2, ge=0, le=1)
    conditioning: Optional[str] = None
    tile_mode: str = ""
    allow_compose_phase: bool = True
    is_intermediate: bool = False
    collect_progress_latents: bool = False
    caption_text: str = Field("", description="text to be overlaid on the image")

    class MaskMode:
        REPLACE = "replace"
        KEEP = "keep"

    def __init__(self, prompt=None, **kwargs):
        # allows `prompt` to be positional
        super().__init__(prompt=prompt, **kwargs)

    @validator("prompt", "negative_prompt", pre=True, always=True)
    def make_into_weighted_prompts(cls, v):
        # if isinstance(v, list):
        #     v = [WeightedPrompt.parse_obj(p) if isinstance(p, dict) else p for p in v]
        if isinstance(v, str):
            v = [WeightedPrompt(text=v)]
        elif isinstance(v, WeightedPrompt):
            v = [v]
        return v

    @validator("prompt", "negative_prompt", always=True)
    def sort_prompts(cls, v):
        if isinstance(v, list):
            v.sort(key=lambda p: p.weight, reverse=True)
        return v

    @validator("negative_prompt", always=True)
    def validate_negative_prompt(cls, v, values):
        if not v:
            model_config = config.MODEL_CONFIG_SHORTCUTS.get(v, None)
            if model_config:
                v = [WeightedPrompt(text=model_config.default_negative_prompt)]
            else:
                v = [WeightedPrompt(text=config.DEFAULT_NEGATIVE_PROMPT)]

        return v

    @validator("prompt_strength", always=True)
    def validate_prompt_strength(cls, v):
        return 7.5 if v is None else v

    @validator("tile_mode", always=True)
    def validate_tile_mode(cls, v):
        if v is True:
            return "xy"

        if v is False:
            return ""

        v = v.lower()
        assert v in ("", "x", "y", "xy")
        return v

    @validator("init_image", "mask_image", always=True)
    def handle_images(cls, v):
        if isinstance(v, str):
            return LazyLoadingImage(filepath=v)

        return v

    @validator("init_image", always=True)
    def set_init_from_control_inputs(cls, v, values):
        if v is None and values.get("control_inputs"):
            for control_input in values["control_inputs"]:
                if control_input.image:
                    return control_input.image

        return v

    @validator("control_inputs", always=True)
    def set_image_from_init_image(cls, v, values):
        v = v or []
        for control_input in v:
            print(control_input)
            if control_input.image is None and control_input.image_raw is None:
                control_input.image = values["init_image"]
        return v

    @validator("mask_image", always=True)
    def validate_mask_image(cls, v, values):
        if v is not None and values["mask_prompt"] is not None:
            raise ValueError("You can only set one of `mask_image` and `mask_prompt`")
        return v

    @validator("mask_prompt", always=True)
    def validate_mask_prompt(cls, v, values):
        if values["init_image"] is None and v:
            raise ValueError(
                "You must set `init_image` if you want to use `mask_prompt`"
            )
        return v

    @validator("model", always=True)
    def set_default_diffusion_model(cls, v):
        if v is None:
            return config.DEFAULT_MODEL

        return v

    @validator("seed", always=True)
    def validate_seed(cls, v):
        return v

    @validator("fix_faces_fidelity", always=True)
    def validate_fix_faces_fidelity(cls, v):
        if v is None:
            return 0.2

        return v

    @validator("sampler_type", pre=True, always=True)
    def validate_sampler_type(cls, v, values):
        from imaginairy.samplers import SamplerName

        if v is None:
            v = config.DEFAULT_SAMPLER

        v = v.lower()

        if values["model"] == "SD-2.0-v" and v == SamplerName.PLMS:
            raise ValueError("PLMS sampler is not supported for SD-2.0-v model.")

        if values["model"] == "edit" and v in (
            SamplerName.PLMS,
            SamplerName.DDIM,
        ):
            raise ValueError(
                "PLMS and DDIM samplers are not supported for pix2pix edit model."
            )
        return v

    @validator("steps", always=True)
    def validate_steps(cls, v, values):
        from imaginairy.samplers import SAMPLER_LOOKUP

        if v is None:
            SamplerCls = SAMPLER_LOOKUP[values["sampler_type"]]
            v = SamplerCls.default_steps

        return int(v)

    @validator("init_image_strength", always=True)
    def validate_init_image_strength(cls, v, values):
        if v is None:
            if values.get("control_inputs"):
                v = 0.0
            elif (
                values.get("outpaint")
                or values.get("mask_image")
                or values.get("mask_prompt")
            ):
                v = 0.0
            else:
                v = 0.2

        return v

    @validator("height", "width", always=True)
    def validate_image_size(cls, v, values):
        from imaginairy.model_manager import get_model_default_image_size

        if v is None:
            v = get_model_default_image_size(values["model"])

        return v

    @validator("caption_text", pre=True, always=True)
    def validate_caption_text(cls, v, values):
        if v is None:
            v = ""

        return v

    @property
    def prompts(self):
        return self.prompt

    @property
    def prompt_text(self):
        if not self.prompt:
            return ""
        if len(self.prompt) == 1:
            return self.prompt[0].text
        return "|".join(str(p) for p in self.prompt)

    @property
    def negative_prompt_text(self):
        if not self.negative_prompt:
            return ""
        if len(self.negative_prompt) == 1:
            return self.negative_prompt[0].text
        return "|".join(str(p) for p in self.negative_prompt)

    def prompt_description(self):
        return (
            f'"{self.prompt_text}" {self.width}x{self.height}px '
            f'negative-prompt:"{self.negative_prompt_text}" '
            f"seed:{self.seed} prompt-strength:{self.prompt_strength} steps:{self.steps} sampler-type:{self.sampler_type} init-image-strength:{self.init_image_strength} model:{self.model}"
        )

    def logging_dict(self):
        """Return a dict of the object but with binary data replaced with reprs."""
        data = self.dict()
        data["init_image"] = repr(self.init_image)
        data["mask_image"] = repr(self.mask_image)
        if self.control_inputs:
            data["control_inputs"] = [repr(ci) for ci in self.control_inputs]
        return data

    def full_copy(self, deep=True, update=None):
        new_prompt = self.copy(
            deep=deep,
            update=update,
        )
        new_prompt = new_prompt.validate(
            dict(
                new_prompt._iter(  # noqa
                    to_dict=False, by_alias=False, exclude_unset=True
                )
            )
        )
        return new_prompt

    def make_concrete_copy(self):
        seed = self.seed if self.seed is not None else random.randint(1, 1_000_000_000)
        return self.full_copy(
            deep=False,
            update={
                "seed": seed,
            },
        )


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

        if result_images:
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
            "prompt": self.prompt.logging_dict(),
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

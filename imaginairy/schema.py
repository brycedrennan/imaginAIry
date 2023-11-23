# pylint: disable=E0213
import base64
import hashlib
import io
import json
import logging
import os.path
import random
from datetime import datetime, timezone
from io import BytesIO
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from imaginairy import config

if TYPE_CHECKING:
    from PIL import Image
else:
    Image = Any


logger = logging.getLogger(__name__)


def save_image_as_base64(image: "Image.Image") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode()


def load_image_from_base64(image_str: str) -> "Image.Image":
    from PIL import Image

    img_bytes = base64.b64decode(image_str)
    return Image.open(io.BytesIO(img_bytes))


class InvalidUrlError(ValueError):
    pass


class LazyLoadingImage:
    """Image file encoded as base64 string."""

    def __init__(
        self, *, filepath=None, url=None, img: Image = None, b64: Optional[str] = None
    ):
        if not filepath and not url and not img and not b64:
            msg = "You must specify a url or filepath or img or base64 string"
            raise ValueError(msg)
        if sum([bool(filepath), bool(url), bool(img), bool(b64)]) > 1:
            raise ValueError("You cannot multiple input methods")

        # validate file exists
        if filepath and not os.path.exists(filepath):
            msg = f"File does not exist: {filepath}"
            raise FileNotFoundError(msg)

        # validate url is valid url
        if url:
            from urllib3.exceptions import LocationParseError
            from urllib3.util import parse_url

            try:
                parsed_url = parse_url(url)
            except LocationParseError:
                raise InvalidUrlError(f"Invalid url: {url}")  # noqa
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
                msg = f"Invalid url: {url}"
                raise InvalidUrlError(msg)

        if b64:
            img = self.load_image_from_base64(b64)

        self._lazy_filepath = filepath
        self._lazy_url = url
        self._img = img

    def __getattr__(self, key):
        if key == "_img":
            #  http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
            raise AttributeError()
        self._load_img()
        return getattr(self._img, key)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

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
                    BytesIO(
                        requests.get(self._lazy_url, stream=True, timeout=60).content
                    )
                )

                logger.debug(
                    f"Loaded input 🖼  of size {self._img.size} from {self._lazy_url}"
                )
            else:
                raise ValueError("You must specify a url or filepath")
            # fix orientation
            self._img = ImageOps.exif_transpose(self._img)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate(value: Any) -> "LazyLoadingImage":
            from PIL import Image, UnidentifiedImageError

            if isinstance(value, cls):
                return value
            if isinstance(value, Image.Image):
                return cls(img=value)
            if isinstance(value, str):
                if "." in value[:1000]:
                    try:
                        return cls(filepath=value)
                    except FileNotFoundError as e:
                        raise ValueError(str(e))  # noqa
                try:
                    return cls(b64=value)
                except UnidentifiedImageError:
                    msg = "base64 string was not recognized as a valid image type"
                    raise ValueError(msg)  # noqa
            if isinstance(value, dict):
                return cls(**value)
            msg = "Image value must be either a LazyLoadingImage, PIL.Image.Image or a Base64 string"
            raise ValueError(msg)

        def handle_b64(value: Any) -> "LazyLoadingImage":
            if isinstance(value, str):
                return cls(b64=value)
            msg = "Image value must be either a LazyLoadingImage, PIL.Image.Image or a Base64 string"
            raise ValueError(msg)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.str_schema(),
                    core_schema.no_info_before_validator_function(
                        handle_b64, core_schema.any_schema()
                    ),
                ]
            ),
            python_schema=core_schema.no_info_before_validator_function(
                validate, core_schema.any_schema()
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
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

    def as_base64(self):
        self._load_img()
        return self.save_image_as_base64(self._img)  # type: ignore

    def as_pillow(self):
        self._load_img()
        return self._img

    def __str__(self):
        return self.as_base64()

    def __repr__(self):
        """human readable representation.

        shows filepath or url if available.
        """
        try:
            return f"<LazyLoadingImage filepath={self._lazy_filepath} url={self._lazy_url}>"
        except Exception as e:  # noqa
            return f"<LazyLoadingImage RENDER EXCEPTION*{e}*>"


class ControlNetInput(BaseModel):
    mode: str
    image: Optional[LazyLoadingImage] = None
    image_raw: Optional[LazyLoadingImage] = None
    strength: int = Field(1, ge=0, le=1000)

    # @field_validator("image", "image_raw", mode="before")
    # def validate_images(cls, v):
    #     if isinstance(v, str):
    #         return LazyLoadingImage(filepath=v)
    #
    #     return v

    @field_validator("image_raw")
    def image_raw_validate(cls, v, info: core_schema.FieldValidationInfo):
        if info.data.get("image") is not None and v is not None:
            raise ValueError("You cannot specify both image and image_raw")

        # if v is None and values.get("image") is None:
        #     raise ValueError("You must specify either image or image_raw")

        return v

    @field_validator("mode")
    def mode_validate(cls, v):
        if v not in config.CONTROLNET_CONFIG_SHORTCUTS:
            valid_modes = list(config.CONTROLNET_CONFIG_SHORTCUTS.keys())
            valid_modes = ", ".join(valid_modes)
            msg = f"Invalid controlnet mode: '{v}'. Valid modes are: {valid_modes}"
            raise ValueError(msg)
        return v


class WeightedPrompt(BaseModel):
    text: str
    weight: float = Field(1, ge=0)

    def __repr__(self):
        return f"{self.weight}*({self.text})"


class ImaginePrompt(BaseModel, protected_namespaces=()):
    prompt: Optional[List[WeightedPrompt]] = Field(default=None, validate_default=True)
    negative_prompt: Optional[List[WeightedPrompt]] = Field(
        default=None, validate_default=True
    )
    prompt_strength: Optional[float] = Field(
        default=7.5, le=10_000, ge=-10_000, validate_default=True
    )
    init_image: Optional[LazyLoadingImage] = Field(
        None, description="base64 encoded image", validate_default=True
    )
    init_image_strength: Optional[float] = Field(
        ge=0, le=1, default=None, validate_default=True
    )
    control_inputs: List[ControlNetInput] = Field(
        default_factory=list, validate_default=True
    )
    mask_prompt: Optional[str] = Field(
        default=None,
        description="text description of the things to be masked",
        validate_default=True,
    )
    mask_image: Optional[LazyLoadingImage] = Field(default=None, validate_default=True)
    mask_mode: Optional[Literal["keep", "replace"]] = "replace"
    mask_modify_original: bool = True
    outpaint: Optional[str] = ""
    model: str = Field(default=config.DEFAULT_MODEL, validate_default=True)
    model_config_path: Optional[str] = None
    sampler_type: str = Field(default=config.DEFAULT_SAMPLER, validate_default=True)
    seed: Optional[int] = Field(default=None, validate_default=True)
    steps: Optional[int] = Field(default=None, validate_default=True)
    height: Optional[int] = Field(None, ge=1, le=100_000, validate_default=True)
    width: Optional[int] = Field(None, ge=1, le=100_000, validate_default=True)
    upscale: bool = False
    fix_faces: bool = False
    fix_faces_fidelity: Optional[float] = Field(0.2, ge=0, le=1, validate_default=True)
    conditioning: Optional[str] = None
    tile_mode: str = ""
    allow_compose_phase: bool = True
    is_intermediate: bool = False
    collect_progress_latents: bool = False
    caption_text: str = Field(
        "", description="text to be overlaid on the image", validate_default=True
    )

    class MaskMode:
        REPLACE = "replace"
        KEEP = "keep"

    def __init__(self, prompt=None, **kwargs):
        # allows `prompt` to be positional
        super().__init__(prompt=prompt, **kwargs)

    @field_validator("prompt", "negative_prompt", mode="before")
    @classmethod
    def make_into_weighted_prompts(cls, v):
        if isinstance(v, str):
            v = [WeightedPrompt(text=v)]
        elif isinstance(v, WeightedPrompt):
            v = [v]
        return v

    @field_validator("prompt", "negative_prompt", mode="after")
    @classmethod
    def must_have_some_weight(cls, v):
        if v:
            total_weight = sum(p.weight for p in v)
            if total_weight == 0:
                raise ValueError("Total weight of prompts cannot be 0")
        return v

    @field_validator("prompt", "negative_prompt", mode="after")
    def sort_prompts(cls, v):
        if isinstance(v, list):
            v.sort(key=lambda p: p.weight, reverse=True)
        return v

    @model_validator(mode="after")
    def validate_negative_prompt(self):
        if self.negative_prompt is None:
            model_config = config.MODEL_CONFIG_SHORTCUTS.get(self.model, None)
            if model_config:
                self.negative_prompt = [
                    WeightedPrompt(text=model_config.default_negative_prompt)
                ]
            else:
                self.negative_prompt = [
                    WeightedPrompt(text=config.DEFAULT_NEGATIVE_PROMPT)
                ]
        return self

    @field_validator("prompt_strength")
    def validate_prompt_strength(cls, v):
        return 7.5 if v is None else v

    @field_validator("tile_mode", mode="before")
    def validate_tile_mode(cls, v):
        valid_tile_modes = ("", "x", "y", "xy")
        if v is True:
            return "xy"

        if v is False or v is None:
            return ""

        if not isinstance(v, str):
            msg = f"Invalid tile_mode: '{v}'. Valid modes are: {valid_tile_modes}"
            raise ValueError(msg)  # noqa

        v = v.lower()
        if v not in valid_tile_modes:
            msg = f"Invalid tile_mode: '{v}'. Valid modes are: {valid_tile_modes}"
            raise ValueError(msg)
        return v

    @field_validator("outpaint", mode="after")
    def validate_outpaint(cls, v):
        from imaginairy.outpaint import outpaint_arg_str_parse

        outpaint_arg_str_parse(v)
        return v

    @field_validator("conditioning", mode="after")
    def validate_conditioning(cls, v):
        from torch import Tensor

        if v is None:
            return v

        if not isinstance(v, Tensor):
            raise ValueError("conditioning must be a torch.Tensor")  # noqa
        return v

    # @field_validator("init_image", "mask_image", mode="after")
    # def handle_images(cls, v):
    #     if isinstance(v, str):
    #         return LazyLoadingImage(filepath=v)
    #
    #     return v

    @model_validator(mode="after")
    def set_init_from_control_inputs(self):
        if self.init_image is None:
            for control_input in self.control_inputs:
                if control_input.image:
                    self.init_image = control_input.image
                    break

        return self

    @field_validator("control_inputs", mode="before")
    def validate_control_inputs(cls, v):
        if v is None:
            v = []
        return v

    @field_validator("control_inputs", mode="after")
    def set_image_from_init_image(cls, v, info: core_schema.FieldValidationInfo):
        v = v or []
        for control_input in v:
            if control_input.image is None and control_input.image_raw is None:
                control_input.image = info.data["init_image"]
        return v

    @field_validator("mask_image")
    def validate_mask_image(cls, v, info: core_schema.FieldValidationInfo):
        if v is not None and info.data.get("mask_prompt") is not None:
            msg = "You can only set one of `mask_image` and `mask_prompt`"
            raise ValueError(msg)
        return v

    @field_validator("mask_prompt", "mask_image", mode="before")
    def validate_mask_prompt(cls, v, info: core_schema.FieldValidationInfo):
        if info.data.get("init_image") is None and v:
            msg = "You must set `init_image` if you want to use a mask"
            raise ValueError(msg)
        return v

    @field_validator("model", mode="before")
    def set_default_diffusion_model(cls, v):
        if v is None:
            return config.DEFAULT_MODEL

        return v

    @field_validator("seed")
    def validate_seed(cls, v):
        return v

    @field_validator("fix_faces_fidelity", mode="before")
    def validate_fix_faces_fidelity(cls, v):
        if v is None:
            return 0.2

        return v

    @field_validator("sampler_type", mode="after")
    def validate_sampler_type(cls, v, info: core_schema.FieldValidationInfo):
        from imaginairy.samplers import SamplerName

        if v is None:
            v = config.DEFAULT_SAMPLER

        v = v.lower()

        if info.data.get("model") == "SD-2.0-v" and v == SamplerName.PLMS:
            raise ValueError("PLMS sampler is not supported for SD-2.0-v model.")

        if info.data.get("model") == "edit" and v in (
            SamplerName.PLMS,
            SamplerName.DDIM,
        ):
            msg = "PLMS and DDIM samplers are not supported for pix2pix edit model."
            raise ValueError(msg)
        return v

    @field_validator("steps")
    def validate_steps(cls, v, info: core_schema.FieldValidationInfo):
        from imaginairy.samplers import SAMPLER_LOOKUP

        if v is None:
            SamplerCls = SAMPLER_LOOKUP[info.data["sampler_type"]]
            v = SamplerCls.default_steps

        return int(v)

    @model_validator(mode="after")
    def validate_init_image_strength(self):
        if self.init_image_strength is None:
            if self.control_inputs:
                self.init_image_strength = 0.0
            elif self.outpaint or self.mask_image or self.mask_prompt:
                self.init_image_strength = 0.0
            else:
                self.init_image_strength = 0.2

        return self

    @field_validator("height", "width")
    def validate_image_size(cls, v, info: core_schema.FieldValidationInfo):
        from imaginairy.model_manager import get_model_default_image_size

        if v is None:
            v = get_model_default_image_size(info.data["model"])

        return v

    @field_validator("caption_text", mode="before")
    def validate_caption_text(cls, v):
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
        data = self.model_dump()
        data["init_image"] = repr(self.init_image)
        data["mask_image"] = repr(self.mask_image)
        if self.control_inputs:
            data["control_inputs"] = [repr(ci) for ci in self.control_inputs]
        return data

    def full_copy(self, deep=True, update=None):
        new_prompt = self.model_copy(
            deep=deep,
            update=update,
        )
        # new_prompt = self.model_validate(new_prompt) doesn't work for some reason https://github.com/pydantic/pydantic/issues/7387
        new_prompt = new_prompt.model_validate(dict(new_prompt))
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
        self.created_at = datetime.now(tz=timezone.utc)
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
            msg = f"Image of type {image_type} not stored. Options are: {self.images.keys()}"
            raise ValueError(msg)

        img.convert("RGB").save(save_path, exif=self._exif())


class SafetyMode:
    STRICT = "strict"
    RELAXED = "relaxed"

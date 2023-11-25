# pylint: disable=E0213
import base64
import hashlib
import io
import json
import random
import logging
import copy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing_extensions import TypedDict
from typing import Any, overload
from urllib3.util import parse_url, Url

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic.dataclasses import dataclass as model_dataclass
from dataclasses import dataclass
from pydantic_core import core_schema
import requests

from imaginairy import config
from PIL import Image, ImageOps


logger = logging.getLogger(__name__)


class InvalidUrlError(ValueError):
    pass


class LazyLoadingImage:
    """Image file encoded as base64 string."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: Image.Image | Path | Url

    @overload
    def __init__(self, *, filepath: str | Path) -> None:
        ...

    @overload
    def __init__(self, *, url: str) -> None:
        ...

    @overload
    def __init__(self, *, img: Image.Image) -> None:
        ...

    @overload
    def __init__(self, *, b64: str) -> None:
        ...

    def __init__(
        self,
        *,
        filepath: str | Path | None = None,
        url: str | None = None,
        img: Image.Image | None = None,
        b64: str | None = None,
    ) -> None:
        match sum(param is not None for param in [filepath, url, img, b64]):
            case 0:
                raise ValueError(
                    "You must specify a url or filepath or img or base64 string"
                )
            case s if s > 1:
                raise ValueError("You cannot multiple input methods")
            case _:
                ...

        if filepath is not None:
            self.data = self._parse_filepath(filepath)

        if url is not None:
            self.data = self._parse_url(url)

        if img is not None:
            self.data = img

        if b64 is not None:
            self.data = self._parse_base64(b64)

    # def __getattr__(self, key: str) -> Any:
    #     if not key == "data":
    #         self._load_data()
    #     return super().__getattribute__(key)

    # def __setstate__(self, state: dict[str, Any]) -> None:
    #     self.__dict__.update(state)

    # def __getstate__(self) -> dict[str, Any]:
    #     return self.__dict__

    def __str__(self) -> str:
        return self.as_base64()

    def __repr__(self) -> str:
        match self.data:
            case Path():
                return f"LazyLoadingImage(filepath={self.data})"
            case Url():
                return f"LazyLoadingImage(url={self.data})"
            case Image.Image():
                return f"LazyLoadingImage(img={self.data})"

    @classmethod
    def __get_pydantic_core_schema__(cls, *_: Any) -> core_schema.CoreSchema:
        def validate(value: Any) -> "LazyLoadingImage":
            if isinstance(value, LazyLoadingImage):
                return value
            raise ValueError("Image value must be a LazyLoadingImage")

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

    @property
    def img(self) -> Image.Image:
        return self._load_data()

    def as_base64(self) -> str:
        buffered = io.BytesIO()
        self.img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode()

    def _parse_filepath(self, filepath: str | Path) -> Path:
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"File does not exist: {filepath}")
        return filepath

    def _parse_url(self, url: str) -> Url:
        from urllib3.exceptions import LocationParseError

        try:
            parsed_url = parse_url(url)
        except LocationParseError:
            raise InvalidUrlError(f"Invalid url: {url}")
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
            msg = f"Invalid url: {url}"
            raise InvalidUrlError(msg)
        return parsed_url

    def _parse_base64(self, b64: str) -> Image.Image:
        img_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(img_bytes))

    def _load_data(self) -> Image.Image:
        match self.data:
            case Image.Image():
                return self._data
            case Path():
                img = Image.open(self.data)
                logger.debug(f"Loaded input 🖼  of size {img.size} from {self.data}")
                self._data = img
                return img
            case Url():
                img = Image.open(
                    BytesIO(
                        requests.get(self.data.url, stream=True, timeout=60).content
                    )
                )
                logger.debug(f"Loaded input 🖼  of size {img.size} from {self.data}")
                self._data = ImageOps.exif_transpose(img)
                return img


class ControlNetInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mode: str
    image: LazyLoadingImage | None = None
    image_raw: LazyLoadingImage | None = None
    strength: int = Field(default=1, ge=0, le=1000)

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
    weight: float = Field(default=1, ge=0)

    def __init__(self, text: "str | WeightedPrompt", weight: float = 1) -> None:
        if isinstance(text, WeightedPrompt):
            weight = text.weight
            text = text.text
        super().__init__(text=text, weight=weight)

    def __bool__(self) -> bool:
        return bool(self.text)

    def __repr__(self) -> str:
        return f"{self.weight}*({self.text})"


class InitImageDict(TypedDict, total=False):
    url: str
    filepath: str | Path
    b64: str
    image: Image.Image
    strength: float


@model_dataclass
class InitImage:
    image: LazyLoadingImage
    strength: float = Field(default=0.2, ge=0, le=1)

    @classmethod
    def parse(cls, init_image: "InitImageDict") -> "InitImage":
        if not "strength" in init_image:
            init_image["strength"] = 0.2

        match init_image:
            case init_image if "image" in init_image:
                image = LazyLoadingImage(img=init_image["image"])
            case init_image if "url" in init_image:
                image = LazyLoadingImage(url=init_image["url"])
            case init_image if "filepath" in init_image:
                image = LazyLoadingImage(filepath=init_image["filepath"])
            case init_image if "b64" in init_image:
                image = LazyLoadingImage(b64=init_image["b64"])
            case _:
                raise ValueError(
                    "You must specify one of url, filepath, b64, image in init_image"
                )

        return cls(image=image, strength=init_image["strength"])


PromptInput = str | WeightedPrompt | list[WeightedPrompt] | list[str]
InitImageInput = InitImage | InitImageDict | LazyLoadingImage | Image.Image | str


class ImaginePromptModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: PromptInput | None = None
    negative_prompt: PromptInput | None = None
    prompt_strength: float = Field(default=7.5, le=50, ge=0)
    init_image: InitImageInput | None = None
    mask_image: LazyLoadingImage | None = None
    control_inputs: list[ControlNetInput] | None = Field(default_factory=list)
    # mask_prompt: str | None = Field(
    #     default=None,
    #     description="text description of the things to be masked",
    #     validate_default=True,
    # )
    # mask_mode: Optional[Literal["keep", "replace"]] = "replace"
    # mask_modify_original: bool = True
    # outpaint: Optional[str] = ""
    model: str = Field(default=config.DEFAULT_MODEL)
    # model_config_path: Optional[str] = None
    sampler_type: str = Field(default=config.DEFAULT_SAMPLER)
    seed: int | None = Field(default=None, ge=1, le=1_000_000_000)
    steps: int = Field(default=30, ge=1, le=1_000)
    height: int | None = Field(default=None, ge=1, le=10_000)
    width: int | None = Field(default=None, ge=1, le=10_000)
    # upscale: bool = False
    # fix_faces: bool = False
    # fix_faces_fidelity: Optional[float] = Field(
    #     default=0.2, ge=0, le=1, validate_default=True
    # )
    # conditioning: Optional[str] = None
    # tile_mode: str = ""
    # allow_compose_phase: bool = True
    # is_intermediate: bool = False
    # collect_progress_latents: bool = False
    # caption_text: str = Field( 
    #     default="",
    #     description="text to be overlaid on the image",
    #     validate_default=True,
    # )

    @field_validator("prompt", "negative_prompt", mode="before")
    @classmethod
    def make_into_weighted_prompts(
        cls, value: str | WeightedPrompt | list[WeightedPrompt] | list[str] | None
    ) -> list[WeightedPrompt]:
        match value:
            case str():
                return [WeightedPrompt(text=value)]
            case WeightedPrompt():
                return [value]
            case list():
                return sorted(
                    [WeightedPrompt(text=p) for p in value],
                    key=lambda p: p.weight,
                    reverse=True,
                )
            case None:
                return [WeightedPrompt(text="")]

    @field_validator("prompt", "negative_prompt", mode="after")
    @classmethod
    def must_have_some_weight(cls, value: list[WeightedPrompt]) -> list[WeightedPrompt]:
        if not any(p.weight for p in value):
            raise ValueError("Total weight of prompts cannot be 0")
        return value

    @field_validator("init_image", mode="after")
    def handle_images(cls, value: InitImageInput | None) -> InitImage | None:
        match value:
            case InitImage():
                return value
            case LazyLoadingImage():
                return InitImage(image=value)
            case str():
                return InitImage(image=LazyLoadingImage(url=value))
            case None:
                return None
            case Image.Image():
                return InitImage(image=LazyLoadingImage(img=value))
            case _:
                return InitImage.parse(value)

    @field_validator("seed")
    def validate_seed(cls, value: int | None) -> int | None:
        if value is None:
            return random.randint(1, 1_000_000_000)
        return value


@dataclass
class ImaginePrompt:
    prompt: list[WeightedPrompt]
    negative_prompt: list[WeightedPrompt]
    prompt_strength: float
    init_image: InitImage | None
    width: int
    height: int
    seed: int
    steps: int
    model: str
    sampler_type: str
    control_inputs: list[ControlNetInput]

    def __init__(
        self,
        prompt: PromptInput | None = None,
        negative_prompt: PromptInput | None = None,
        prompt_strength: float = 7.5,
        init_image: InitImageInput | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int = 30,
        model: str = config.DEFAULT_MODEL,
        sampler_type: str = config.DEFAULT_SAMPLER,
        control_inputs: list[ControlNetInput] | None = None,
    ) -> None:
        data = ImaginePromptModel(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_strength=prompt_strength,
            init_image=init_image,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            model=model,
            sampler_type=sampler_type,
            control_inputs=control_inputs,
        )
        self.__dict__.update(data)

    # @model_validator(mode="after")
    # def validate_negative_prompt(self):
    #     if self.negative_prompt is None:
    #         model_config = config.MODEL_CONFIG_SHORTCUTS.get(self.model, None)
    #         if model_config:
    #             self.negative_prompt = [
    #                 WeightedPrompt(text=model_config.default_negative_prompt)
    #             ]
    #         else:
    #             self.negative_prompt = [
    #                 WeightedPrompt(text=config.DEFAULT_NEGATIVE_PROMPT)
    #             ]
    #     return self

    # @field_validator("tile_mode", mode="before")
    # def validate_tile_mode(cls, v):
    #     valid_tile_modes = ("", "x", "y", "xy")
    #     if v is True:
    #         return "xy"

    #     if v is False or v is None:
    #         return ""

    #     if not isinstance(v, str):
    #         msg = f"Invalid tile_mode: '{v}'. Valid modes are: {valid_tile_modes}"
    #         raise ValueError(msg)  # noqa

    #     v = v.lower()
    #     if v not in valid_tile_modes:
    #         msg = f"Invalid tile_mode: '{v}'. Valid modes are: {valid_tile_modes}"
    #         raise ValueError(msg)
    #     return v

    # @field_validator("outpaint", mode="after")
    # def validate_outpaint(cls, v):
    #     from imaginairy.outpaint import outpaint_arg_str_parse

    #     outpaint_arg_str_parse(v)
    #     return v

    # @field_validator("conditioning", mode="after")
    # def validate_conditioning(cls, v):
    #     from torch import Tensor

    #     if v is None:
    #         return v

    #     if not isinstance(v, Tensor):
    #         raise ValueError("conditioning must be a torch.Tensor")  # noqa
    #     return v

    # @model_validator(mode="after")
    # def set_init_from_control_inputs(self):
    #     if self.init_image is None:
    #         for control_input in self.control_inputs:
    #             if control_input.image:
    #                 self.init_image = control_input.image
    #                 break

    #     return self

    # @field_validator("control_inputs", mode="before")
    # def validate_control_inputs(cls, v):
    #     if v is None:
    #         v = []
    #     return v

    # @field_validator("control_inputs", mode="after")
    # def set_image_from_init_image(cls, v, info: core_schema.FieldValidationInfo):
    #     v = v or []
    #     for control_input in v:
    #         if control_input.image is None and control_input.image_raw is None:
    #             control_input.image = info.data["init_image"]
    #     return v

    # @field_validator("mask_image")
    # def validate_mask_image(cls, v, info: core_schema.FieldValidationInfo):
    #     if v is not None and info.data.get("mask_prompt") is not None:
    #         msg = "You can only set one of `mask_image` and `mask_prompt`"
    #         raise ValueError(msg)
    #     return v

    # @field_validator("mask_prompt", "mask_image", mode="before")
    # def validate_mask_prompt(cls, v, info: core_schema.FieldValidationInfo):
    #     if info.data.get("init_image") is None and v:
    #         msg = "You must set `init_image` if you want to use a mask"
    #         raise ValueError(msg)
    #     return v

    # @field_validator("model", mode="before")
    # def set_default_diffusion_model(cls, v):
    #     if v is None:
    #         return config.DEFAULT_MODEL

    #     return v

    # @field_validator("fix_faces_fidelity", mode="before")
    # def validate_fix_faces_fidelity(cls, v):
    #     if v is None:
    #         return 0.2

    #     return v

    # @field_validator("sampler_type", mode="after")
    # def validate_sampler_type(cls, v, info: core_schema.FieldValidationInfo):
    #     from imaginairy.samplers import SamplerName

    #     if v is None:
    #         v = config.DEFAULT_SAMPLER

    #     v = v.lower()

    #     if info.data.get("model") == "SD-2.0-v" and v == SamplerName.PLMS:
    #         raise ValueError("PLMS sampler is not supported for SD-2.0-v model.")

    #     if info.data.get("model") == "edit" and v in (
    #         SamplerName.PLMS,
    #         SamplerName.DDIM,
    #     ):
    #         msg = "PLMS and DDIM samplers are not supported for pix2pix edit model."
    #         raise ValueError(msg)
    #     return v

    # @field_validator("steps")
    # def validate_steps(cls, v, info: core_schema.FieldValidationInfo):
    #     from imaginairy.samplers import SAMPLER_LOOKUP

    #     if v is None:
    #         SamplerCls = SAMPLER_LOOKUP[info.data["sampler_type"]]
    #         v = SamplerCls.default_steps

    #     return int(v)

    # @model_validator(mode="after")
    # def validate_init_image_strength(self):
    #     if self.init_image_strength is None:
    #         if self.control_inputs:
    #             self.init_image_strength = 0.0
    #         elif self.outpaint or self.mask_image or self.mask_prompt:
    #             self.init_image_strength = 0.0
    #         else:
    #             self.init_image_strength = 0.2

    #     return self

    # @field_validator("height", "width")
    # def validate_image_size(cls, v, info: core_schema.FieldValidationInfo):
    #     from imaginairy.model_manager import get_model_default_image_size

    #     if v is None:
    #         v = get_model_default_image_size(info.data["model"])

    #     return v

    # @field_validator("caption_text", mode="before")
    # def validate_caption_text(cls, v):
    #     if v is None:
    #         v = ""

    #     return v

    @property
    def prompt_text(self) -> str:
        if len(self.prompt) == 1:
            return self.prompt[0].text
        return "|".join(str(p) for p in self.prompt)

    @property
    def negative_prompt_text(self) -> str:
        if len(self.negative_prompt) == 1:
            return self.negative_prompt[0].text
        return "|".join(str(p) for p in self.negative_prompt)

    @property
    def init_image_strength(self) -> float:
        return self.init_image.strength if self.init_image else 0

    def prompt_description(self) -> str:
        return (
            f'"{self.prompt_text}" {self.width}x{self.height}px '
            f'negative-prompt:"{self.negative_prompt_text}" '
            f"seed:{self.seed} prompt-strength:{self.prompt_strength} steps:{self.steps} sampler-type:{self.sampler_type} init-image-strength:{self.init_image_strength} model:{self.model}"
        )

    # def logging_dict(self):
    #     """Return a dict of the object but with binary data replaced with reprs."""
    #     data = self.model_dump()
    #     data["init_image"] = repr(self.init_image)
    #     data["mask_image"] = repr(self.mask_image)
    #     if self.control_inputs:
    #         data["control_inputs"] = [repr(ci) for ci in self.control_inputs]
    #     return data

    def full_copy(self, deep: bool = True, update: dict[str, Any] | None = None):
        """Return a copy of the object while perfoming an update."""
        if update is None:
            update = {}
        data = copy.deepcopy(self) if deep else copy.copy(self)
        data.__dict__.update(update)
        return data

    def make_concrete_copy(self):
        return self.full_copy(deep=True)


# from dataclasses import dataclass


# @dataclass
# class ImaginePrompt:
#     prompts: list[WeightedPrompt]
#     negative_prompts: list[WeightedPrompt]
#     prompt_strength: float
#     init_image: LazyLoadingImage | None
#     init_image_strength: float
#     control_inputs: list[ControlNetInput]
#     mask_prompt: str | None
#     mask_image: Optional[LazyLoadingImage] = Field(default=None, validate_default=True)
#     mask_mode: Optional[Literal["keep", "replace"]] = "replace"
#     mask_modify_original: bool = True
#     outpaint: Optional[str] = ""
#     model: str = ""
#     model_config_path: Optional[str] = None
#     sampler_type: str = ""
#     seed: int | None = None
#     steps: int | None = None
#     height: int | None = None
#     width: int | None = None
#     upscale: bool = False
#     fix_faces: bool = False
#     fix_faces_fidelity: float = 0.2
#     conditioning: Optional[str] = None
#     tile_mode: str = ""
#     allow_compose_phase: bool = True
#     is_intermediate: bool = False
#     collect_progress_latents: bool = False
#     caption_text: str = ""

#     def __init__(self, prompt: list[WeightedPrompt] | str):
#         model = ImaginePromptModel(prompt=prompt)
#         self.__dict__.update(model.model_dump())


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

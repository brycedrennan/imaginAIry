# pylint: disable=E0213
import base64
import copy
import hashlib
import io
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, overload

import requests
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass as model_dataclass
from pydantic_core import core_schema
from typing_extensions import TypedDict
from urllib3.util import Url, parse_url

from imaginairy import config
from imaginairy.model_manager import get_model_default_image_size  # type: ignore

logger = logging.getLogger(__name__)


class InvalidUrlError(ValueError):
    pass


class LazyLoadingImage:
    """Image file encoded as base64 string."""

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
    config: config.ControlNetConfig | str
    image: LazyLoadingImage | Path | str | None = None
    image_raw: LazyLoadingImage | Path | str | None = None
    strength: int = Field(default=1, ge=0, le=1000)

    @field_validator("image", "image_raw", mode="before")
    def validate_images(
        cls, value: LazyLoadingImage | str | None
    ) -> LazyLoadingImage | None:
        match value:
            case LazyLoadingImage():
                return value
            case None:
                return None
            case str():
                return LazyLoadingImage(url=value)


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
class InitImageConfig:
    image: LazyLoadingImage
    strength: float = Field(default=0.2, ge=0, le=1)

    @classmethod
    def parse(cls, init_image: "InitImageDict") -> "InitImageConfig":
        if "strength" not in init_image:
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


InitImageInput = InitImageConfig | InitImageDict | LazyLoadingImage | Image.Image | str


class MaskMode(str, Enum):  # replace with strEnum when python >=3.11
    KEEP = "keep"
    REPLACE = "replace"


@model_dataclass
class MaskConfig:
    image: LazyLoadingImage | None = None
    prompt: str | None = None
    mode: MaskMode = MaskMode.REPLACE
    modify_original: bool = True


MaskInput = MaskConfig | LazyLoadingImage | str


class SamplerType(str, Enum):  # replace with strEnum when python >=3.11
    DDIM = "ddim"
    DPMv2 = "dpmv2"


@model_dataclass
class FaceFixerConfig:
    enabled: bool = False
    fidelity: float = Field(default=0.2, ge=0, le=1, validate_default=True)


PromptInput = str | WeightedPrompt | list[WeightedPrompt] | list[str]


class TileMode(str, Enum):  # replace with strEnum when python >=3.11
    X = "x"
    Y = "y"
    XY = "xy"


class ImaginePromptModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: PromptInput | None = None
    negative_prompt: PromptInput | None = None
    prompt_strength: float = Field(default=7.5, le=50, ge=0)
    init_image: InitImageInput | None = None
    mask: MaskInput | None = None
    control_inputs: list[ControlNetInput] | None = Field(default_factory=list)
    outpaint: str | None = None
    model: str = Field(default=config.DEFAULT_MODEL)
    sampler_type: SamplerType | str = config.DEFAULT_SAMPLER
    seed: int | None = Field(default=None, ge=1, le=1_000_000_000)
    steps: int = Field(default=30, ge=1, le=1_000)
    height: int | None = Field(default=None, ge=1, le=10_000)
    width: int | None = Field(default=None, ge=1, le=10_000)
    upscale: bool = False
    face_fixer: FaceFixerConfig | float | bool | None = None
    tiling: TileMode | str | bool | None = None
    caption_text: str | None = None

    @field_validator("prompt", "negative_prompt", mode="before")
    def make_into_weighted_prompts(
        cls,
        value: str | WeightedPrompt | list[WeightedPrompt] | list[str] | None,
        info: ValidationInfo,
    ) -> list[WeightedPrompt]:
        match value:
            case str():
                if info.field_name == "negative_prompt" and value == "":
                    return [WeightedPrompt(text=config.DEFAULT_NEGATIVE_PROMPT)]
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
    def must_have_some_weight(cls, value: list[WeightedPrompt]) -> list[WeightedPrompt]:
        if not any(p.weight for p in value):
            raise ValueError("Total weight of prompts cannot be 0")
        return value

    @field_validator("init_image", mode="after")
    def handle_images(cls, value: InitImageInput | None) -> InitImageConfig | None:
        match value:
            case InitImageConfig():
                return value
            case LazyLoadingImage():
                return InitImageConfig(image=value)
            case str():
                return InitImageConfig(image=LazyLoadingImage(url=value))
            case None:
                return None
            case Image.Image():
                return InitImageConfig(image=LazyLoadingImage(img=value))
            case _:
                return InitImageConfig.parse(value)

    @field_validator("seed")
    def validate_seed(cls, value: int | None) -> int | None:
        if value is None:
            return random.randint(1, 1_000_000_000)
        return value

    @field_validator("tiling", mode="before")
    def validate_tiling(
        cls, value: TileMode | str | bool | None
    ) -> TileMode | str | bool | None:
        match value:
            case True:
                return TileMode.XY
            case str():
                return TileMode(value.lower())
            case TileMode():
                return value
            case _:
                return None

    @field_validator("outpaint", mode="after")
    def validate_outpaint(cls, value: str | None) -> str | None:
        from imaginairy.outpaint import outpaint_arg_str_parse  # type: ignore

        outpaint_arg_str_parse(value)  # TODO improve this
        return value

    @field_validator("height", "width")
    def validate_image_size(
        cls, value: int | None, info: core_schema.FieldValidationInfo
    ) -> int:
        return value or get_model_default_image_size(info.data["model"])

    @field_validator("face_fixer", mode="before")
    def validate_face_fixer(
        cls, value: FaceFixerConfig | float | bool | None
    ) -> FaceFixerConfig | None:
        match value:
            case FaceFixerConfig():
                return value
            case float():
                return FaceFixerConfig(enabled=True, fidelity=value)
            case bool():
                return FaceFixerConfig(enabled=value)
            case None:
                return None
            case _:
                raise ValueError("Invalid face_fixer value")

    @field_validator("caption_text", mode="before")
    def validate_caption_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @field_validator("control_inputs", mode="before")
    def validate_control_inputs(
        cls, value: list[ControlNetInput] | None
    ) -> list[ControlNetInput]:
        if value is None:
            return []
        return value

    @field_validator("mask", mode="after")
    def validate_mask(
        cls, value: MaskConfig | LazyLoadingImage | str | None
    ) -> MaskConfig | None:
        match value:
            case MaskConfig():
                return value
            case LazyLoadingImage():
                return MaskConfig(image=value)
            case str():
                return MaskConfig(image=LazyLoadingImage(url=value))
            case None:
                return None

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


@dataclass
class ImaginePrompt:
    prompt: list[WeightedPrompt]
    negative_prompt: list[WeightedPrompt]
    prompt_strength: float
    init_image: InitImageConfig | None
    width: int
    height: int
    seed: int
    steps: int
    model: str
    model_config: Path  # ?: what is this?
    sampler_type: SamplerType
    control_inputs: list[ControlNetInput]
    outpaint: str  # TODO: implement OutpaintConfig class
    face_fixer: FaceFixerConfig
    tiling: TileMode
    caption_text: str | None

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
        outpaint: str = "",
        face_fixer: FaceFixerConfig | float | bool | None = None,
        tiling: TileMode | str | bool | None = None,
        caption_text: str | None = None,
    ) -> None:
        """
        Args:
            prompt:
                Text prompt(s) to condition the model on. Can be a single string
                or a list of strings.
            negative_prompt:
                Text prompt(s) to condition the model on. Can be a single string
                or a list of strings.
            prompt_strength:
                How much to weigh the prompt when sampling. Higher values mean
                more influence, but can also lead to more artifacts.
            init_image:
                Image to initialize the model with. Can be a filepath, url,
                base64 string, PIL Image, or LazyLoadingImage.
            width:
                Width of the generated image in pixels.
            height:
                Height of the generated image in pixels.
            seed:
                Random seed to use for sampling.
            steps:
                Number of sampling steps to perform.
            model:
                Name of the model to use. See the list of available models in
                imaginairy.config.
            sampler_type:
                Name of the sampler to use. See the list of available samplers in config.py.
            control_inputs:
                List of ControlNetInput objects to use for conditioning.
            outpaint:
                Instructions for outpainting #TODO: implement OutpaintConfig class
            face_fixer:
                Config for face fixing algorithm (CodeFormer)
            tiling:
                Allow generating images with that can be tiled in x, y, or both directions.
            caption_text:
                Text to add on top of the generated image.
        """
        self._model = ImaginePromptModel(
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
            outpaint=outpaint,
            face_fixer=face_fixer,
            tiling=tiling,
            caption_text=caption_text,
        )
        self.__dict__.update(self._model)

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

    def full_copy(self, deep: bool = True, update: dict[str, Any] | None = None):
        """Return a copy of the object while perfoming an update."""
        if update is None:
            update = {}
        data = copy.deepcopy(self) if deep else copy.copy(self)
        data.__dict__.update(update)
        return data

    def make_concrete_copy(self):
        return self.full_copy(deep=True)


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
        img: Image.Image,
        prompt: ImaginePrompt,
        is_nsfw: bool = False,
        safety_score: float = 1.0,
        upscaled_img: Image.Image | None = None,
        modified_original: Image.Image | None = None,
        mask_binary: Image.Image | None = None,
        mask_grayscale: Image.Image | None = None,
        result_images: list[Image.Image] | None = None,
        timings: dict[str, float] | None = None,
        progress_latents: list[Image.Image] | None = None,
    ):
        import torch

        from imaginairy.img_utils import model_latent_to_pillow_img  # type: ignore
        from imaginairy.img_utils import torch_img_to_pillow_img
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
            for img_type, r_img in result_images.items():  # type: ignore
                if isinstance(r_img, torch.Tensor):
                    if r_img.shape[1] == 4:
                        r_img = model_latent_to_pillow_img(r_img)  # type: ignore
                    else:
                        r_img = torch_img_to_pillow_img(r_img)
                self.images[img_type] = r_img  # type: ignore

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

    def md5(self) -> str:
        return hashlib.md5(self.img.tobytes()).hexdigest()  # type: ignore

    def metadata_dict(self) -> dict[str, ImaginePrompt]:
        return {"prompt": self.prompt}

    def timings_str(self) -> str:
        if not self.timings:
            return ""
        return " ".join(f"{k}:{v:.2f}s" for k, v in self.timings.items())

    def _exif(self) -> Image.Exif:
        exif = Image.Exif()
        exif[ExifCodes.ImageDescription] = self.prompt.prompt_description()
        exif[ExifCodes.UserComment] = json.dumps(self.metadata_dict())  # type: ignore
        # help future web scrapes not ingest AI generated art
        sd_version = self.prompt.model
        if len(sd_version) > 20:
            sd_version = "custom weights"
        exif[ExifCodes.Software] = f"Imaginairy / Stable Diffusion {sd_version}"
        exif[ExifCodes.DateTime] = self.created_at.isoformat(sep=" ")[:19]
        exif[ExifCodes.HostComputer] = f"{self.torch_backend}:{self.hardware_name}"
        return exif

    def save(self, save_path: Path | str, image_type: str = "generated") -> None:
        img = self.images.get(image_type)
        if img is None:
            msg = f"Image of type {image_type} not stored. Options are: {self.images.keys()}"
            raise ValueError(msg)

        img.convert("RGB").save(save_path, exif=self._exif())


class SafetyMode(str, Enum):
    STRICT = "strict"
    RELAXED = "relaxed"

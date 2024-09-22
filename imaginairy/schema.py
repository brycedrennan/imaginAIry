"""Classes for image generation and manipulation"""

# pylint: disable=E0213
import base64
import hashlib
import io
import json
import logging
import os.path
import random
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any, List, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema
from typing_extensions import Self

from imaginairy import config

if TYPE_CHECKING:
    from pathlib import Path  # noqa

    from PIL import Image


logger = logging.getLogger(__name__)


class InvalidUrlError(ValueError):
    pass


class LazyLoadingImage:
    """
    A class representing an image that can be lazily loaded from various sources.

    This class supports loading an image from a filepath, URL, a PIL Image object,
    or a base64 encoded string. The image is only loaded into memory when it's
    accessed, not at the time of object creation. If multiple sources are provided,
    an error is raised. The class also provides functionality to convert the image
    to a base64 string and to access it as a PIL Image object.

    Attributes:
        _lazy_filepath (str): Path to the image file, if provided.
        _lazy_url (str): URL of the image, if provided.
        _img (Image.Image): PIL Image object, if provided.

    Methods:
        _load_img: Lazily loads the image from the specified source.
        as_base64: Returns the image encoded as a base64 string.
        as_pillow: Returns the image as a PIL Image object.
        save_image_as_base64: Static method to convert a PIL Image to a base64 string.
        load_image_from_base64: Static method to load an image from a base64 string.
        __get_pydantic_core_schema__: Class method for Pydantic schema generation.

    """

    def __init__(
        self,
        *,
        filepath: str | None = None,
        url: str | None = None,
        img: "Image.Image | None" = None,
        b64: str | None = None,
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
        return self.save_image_as_base64(self._img)

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


class ControlInput(BaseModel):
    """
    A Pydantic model representing the input control parameters for an operation,
    typically involving image processing.

    This model includes parameters such as the operation mode, the image to be processed,
    an alternative raw image, and a strength parameter. It validates these parameters to
    ensure they meet specific criteria, such as the mode being one of the predefined valid modes
    and ensuring that both 'image' and 'image_raw' are not provided simultaneously.

    Attributes:
        mode (str): The operation mode, which must be one of the predefined valid modes.
        image (LazyLoadingImage, optional): An instance of LazyLoadingImage to be processed.
                                            Defaults to None.
        image_raw (LazyLoadingImage, optional): An alternative raw image instance of
                                                LazyLoadingImage. Defaults to None.
        strength (float): A float value representing the strength of the operation, must be
                          between 0 and 1000 (inclusive). Defaults to 1.

    Methods:
        image_raw_validate: Validates that either 'image' or 'image_raw' is provided,
                            but not both.
        mode_validate: Validates that the 'mode' attribute is one of the predefined valid
                       modes in the configuration.

    Raises:
        ValueError: Raised if both 'image' and 'image_raw' are specified, or if the
                    'mode' is not a valid mode.
    """

    mode: str
    image: LazyLoadingImage | None = None
    image_raw: LazyLoadingImage | None = None
    strength: float = Field(1, ge=0, le=1000)

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
        if v not in config.CONTROL_CONFIG_SHORTCUTS:
            valid_modes = list(config.CONTROL_CONFIG_SHORTCUTS.keys())
            valid_modes = ", ".join(valid_modes)
            msg = f"Invalid controlnet mode: '{v}'. Valid modes are: {valid_modes}"
            raise ValueError(msg)
        return v


class WeightedPrompt(BaseModel):
    """
    Represents a prompt with an associated weight.

    This class is used to define a text prompt with a corresponding numerical weight,
    indicating the significance or influence of the prompt in a given context, such as
    in image generation or text processing tasks.

    Attributes:
        text (str): The textual content of the prompt.
        weight (float): A numerical weight associated with the prompt. Defaults to 1.
                        The weight must be greater than or equal to 0.

    Methods:
        __repr__: Returns a string representation of the WeightedPrompt instance,
                  formatted as 'weight*(text)'.
    """

    text: str
    weight: float = Field(1, ge=0)

    def __repr__(self):
        return f"{self.weight}*({self.text})"


class MaskMode(str, Enum):
    REPLACE = "replace"
    KEEP = "keep"


MaskInput = MaskMode | str
PromptInput = str | WeightedPrompt | list[WeightedPrompt] | list[str] | None
InpaintMethod = Literal["finetune", "control"]


class ImaginePrompt(BaseModel, protected_namespaces=()):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt: List[WeightedPrompt] = Field(default=None, validate_default=True)  # type: ignore
    negative_prompt: List[WeightedPrompt] = Field(
        default_factory=list, validate_default=True
    )
    prompt_strength: float = Field(default=7.5, le=50, ge=-50, validate_default=True)
    init_image: LazyLoadingImage | None = Field(
        None, description="base64 encoded image", validate_default=True
    )
    init_image_strength: float | None = Field(
        ge=0, le=1, default=None, validate_default=True
    )
    image_prompt: List[LazyLoadingImage] | None = Field(None, validate_default=True)
    image_prompt_strength: float = Field(ge=0, le=1, default=0.0)
    control_inputs: List[ControlInput] = Field(
        default_factory=list, validate_default=True
    )
    mask_prompt: str | None = Field(
        default=None,
        description="text description of the things to be masked",
        validate_default=True,
    )
    mask_image: LazyLoadingImage | None = Field(default=None, validate_default=True)
    mask_mode: MaskMode = MaskMode.REPLACE
    mask_modify_original: bool = True
    outpaint: str | None = ""
    model_weights: config.ModelWeightsConfig = Field(  # type: ignore
        default=config.DEFAULT_MODEL_WEIGHTS, validate_default=True
    )
    solver_type: str = Field(default=config.DEFAULT_SOLVER, validate_default=True)
    seed: int | None = Field(default=None, validate_default=True)
    steps: int = Field(validate_default=True)
    size: tuple[int, int] = Field(validate_default=True)
    upscale: bool = False
    fix_faces: bool = False
    fix_faces_fidelity: float | None = Field(0.5, ge=0, le=1, validate_default=True)
    conditioning: str | None = None
    tile_mode: str = ""
    allow_compose_phase: bool = True
    is_intermediate: bool = False
    collect_progress_latents: bool = False
    caption_text: str = Field(
        "", description="text to be overlaid on the image", validate_default=True
    )
    composition_strength: float = Field(ge=0, le=1, validate_default=True)
    inpaint_method: InpaintMethod = "finetune"

    def __init__(
        self,
        prompt: PromptInput = "",
        *,
        negative_prompt: PromptInput = None,
        prompt_strength: float | None = 7.5,
        init_image: LazyLoadingImage | None = None,
        init_image_strength: float | None = None,
        image_prompt: LazyLoadingImage | List[LazyLoadingImage] | None = None,
        image_prompt_strength: float | None = 0.35,
        control_inputs: List[ControlInput] | None = None,
        mask_prompt: str | None = None,
        mask_image: LazyLoadingImage | None = None,
        mask_mode: MaskInput = MaskMode.REPLACE,
        mask_modify_original: bool = True,
        outpaint: str | None = "",
        model_weights: str | config.ModelWeightsConfig = config.DEFAULT_MODEL_WEIGHTS,
        solver_type: str = config.DEFAULT_SOLVER,
        seed: int | None = None,
        steps: int | None = None,
        size: int | str | tuple[int, int] | None = None,
        upscale: bool = False,
        fix_faces: bool = False,
        fix_faces_fidelity: float | None = 0.2,
        conditioning: str | None = None,
        tile_mode: str = "",
        allow_compose_phase: bool = True,
        is_intermediate: bool = False,
        collect_progress_latents: bool = False,
        caption_text: str = "",
        composition_strength: float | None = 0.5,
        inpaint_method: InpaintMethod = "finetune",
    ):
        if image_prompt and not isinstance(image_prompt, list):
            image_prompt = [image_prompt]

        if not image_prompt_strength:
            image_prompt_strength = 0.35

        super().__init__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_strength=prompt_strength,
            init_image=init_image,
            init_image_strength=init_image_strength,
            image_prompt=image_prompt,
            image_prompt_strength=image_prompt_strength,
            control_inputs=control_inputs,
            mask_prompt=mask_prompt,
            mask_image=mask_image,
            mask_mode=mask_mode,
            mask_modify_original=mask_modify_original,
            outpaint=outpaint,
            model_weights=model_weights,
            solver_type=solver_type,
            seed=seed,
            steps=steps,
            size=size,
            upscale=upscale,
            fix_faces=fix_faces,
            fix_faces_fidelity=fix_faces_fidelity,
            conditioning=conditioning,
            tile_mode=tile_mode,
            allow_compose_phase=allow_compose_phase,
            is_intermediate=is_intermediate,
            collect_progress_latents=collect_progress_latents,
            caption_text=caption_text,
            composition_strength=composition_strength,
            inpaint_method=inpaint_method,
        )
        self._default_negative_prompt = None

    @field_validator("prompt", "negative_prompt", mode="before")
    def make_into_weighted_prompts(
        cls,
        value: PromptInput,
    ) -> list[WeightedPrompt]:
        match value:
            case None:
                return []

            case str():
                if value is not None:
                    return [WeightedPrompt(text=value)]
                else:
                    return []
            case WeightedPrompt():
                return [value]
            case list():
                if all(isinstance(item, str) for item in value):
                    return [WeightedPrompt(text=str(p)) for p in value]
                elif all(isinstance(item, WeightedPrompt) for item in value):
                    return cast(List[WeightedPrompt], value)
        raise ValueError("Invalid prompt input")

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

    @property
    def default_negative_prompt(self):
        default_negative_prompt = config.DEFAULT_NEGATIVE_PROMPT
        if self.model_weights:
            default_negative_prompt = self.model_weights.defaults.get(
                "negative_prompt", default_negative_prompt
            )
        return default_negative_prompt

    @model_validator(mode="after")
    def validate_negative_prompt(self):
        if self.negative_prompt == []:
            self.negative_prompt = [WeightedPrompt(text=self.default_negative_prompt)]

        return self

    @field_validator("prompt_strength", mode="before")
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
        from imaginairy.utils.outpaint import outpaint_arg_str_parse

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

    @model_validator(mode="before")
    @classmethod
    def set_default_composition_strength(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        comp_strength = data.get("composition_strength")
        default_comp_strength = 0.5
        if comp_strength is None:
            model_weights = data.get("model_weights")
            if isinstance(model_weights, config.ModelWeightsConfig):
                default_comp_strength = model_weights.defaults.get(
                    "composition_strength", default_comp_strength
                )
            data["composition_strength"] = default_comp_strength

        return data

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

    @model_validator(mode="before")
    def resolve_model_weights(cls, data: Any):
        if not isinstance(data, dict):
            return data

        model_weights = data.get("model_weights")
        if model_weights is None:
            model_weights = config.DEFAULT_MODEL_WEIGHTS
        from imaginairy.utils.model_manager import resolve_model_weights_config

        should_use_inpainting = bool(
            data.get("mask_image") or data.get("mask_prompt") or data.get("outpaint")
        )
        should_use_inpainting_weights = (
            should_use_inpainting and data.get("inpaint_method") == "finetune"
        )
        model_weights_config = resolve_model_weights_config(
            model_weights=model_weights,
            default_model_architecture=None,
            for_inpainting=should_use_inpainting_weights,
        )
        data["model_weights"] = model_weights_config

        return data

    @field_validator("seed")
    def validate_seed(cls, v):
        return v

    @field_validator("fix_faces_fidelity", mode="before")
    def validate_fix_faces_fidelity(cls, v):
        if v is None:
            return 0.5

        return v

    @field_validator("solver_type", mode="after")
    def validate_solver_type(cls, v, info: core_schema.FieldValidationInfo):
        from imaginairy.samplers import SolverName

        if v is None:
            v = config.DEFAULT_SOLVER

        v = v.lower()

        if info.data.get("model") == "edit" and v in (
            SolverName.PLMS,
            SolverName.DDIM,
        ):
            msg = "PLMS and DDIM solvers are not supported for pix2pix edit model."
            raise ValueError(msg)
        return v

    @field_validator("steps", mode="before")
    def validate_steps(cls, v, info: core_schema.FieldValidationInfo):
        model_weights = info.data.get("model_weights")

        # Try to get steps from model weights defaults
        if (
            v is None
            and model_weights
            and isinstance(model_weights, config.ModelWeightsConfig)
        ):
            v = model_weights.defaults.get("steps")

        # If not found in model weights, try model architecture defaults
        if v is None and model_weights and model_weights.architecture:
            v = model_weights.architecture.defaults.get("steps")

        # If still not found, use solver-specific defaults
        if v is None:
            solver_type = info.data.get("solver_type", "ddim").lower()
            steps_lookup = {"ddim": 50, "dpmpp": 20}
            v = steps_lookup.get(
                solver_type, 50
            )  # Default to 50 if solver not recognized

        try:
            return int(v)
        except (OverflowError, TypeError) as e:
            raise ValueError("Steps must be an integer") from e

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

    @field_validator("size", mode="before")
    def validate_image_size(cls, v, info: core_schema.FieldValidationInfo):
        from imaginairy.utils.model_manager import get_model_default_image_size
        from imaginairy.utils.named_resolutions import normalize_image_size

        if v is None:
            v = get_model_default_image_size(info.data["model_weights"].architecture)

        width, height = normalize_image_size(v)

        return width, height

    @field_validator("size", mode="after")
    def validate_image_size_after(cls, v, info: core_schema.FieldValidationInfo):
        width, height = v
        min_size = 8
        max_size = 100_000
        if not min_size <= width <= max_size:
            msg = f"Width must be between {min_size} and {max_size}. Got: {width}"
            raise ValueError(msg)

        if not min_size <= height <= max_size:
            msg = f"Height must be between {min_size} and {max_size}. Got: {height}"
            raise ValueError(msg)
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
    def prompt_text(self) -> str:
        if not self.prompt:
            return ""
        if len(self.prompt) == 1:
            return self.prompt[0].text
        return "|".join(str(p) for p in self.prompt)

    @property
    def negative_prompt_text(self) -> str:
        if not self.negative_prompt:
            return ""
        if len(self.negative_prompt) == 1:
            return self.negative_prompt[0].text
        return "|".join(str(p) for p in self.negative_prompt)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def aspect_ratio(self) -> str:
        from imaginairy.utils.img_utils import aspect_ratio

        return aspect_ratio(width=self.width, height=self.height)

    @property
    def should_use_inpainting(self) -> bool:
        return bool(self.outpaint or self.mask_image or self.mask_prompt)

    @property
    def should_use_inpainting_weights(self) -> bool:
        return self.should_use_inpainting and self.inpaint_method == "finetune"

    @property
    def model_architecture(self) -> config.ModelArchitecture:
        return self.model_weights.architecture

    def prompt_description(self):
        if self.negative_prompt_text == self.default_negative_prompt:
            neg_prompt = "DEFAULT-NEGATIVE-PROMPT"
        else:
            neg_prompt = f'"{self.negative_prompt_text}"'

        from termcolor import colored

        prompt_text = colored(self.prompt_text, "green")

        return (
            f'"{prompt_text}"\n'
            "    "
            f"negative-prompt:{neg_prompt}\n"
            "    "
            f"size:{self.width}x{self.height}px-({self.aspect_ratio}) "
            f"seed:{self.seed} "
            f"prompt-strength:{self.prompt_strength} "
            f"steps:{self.steps} solver-type:{self.solver_type} "
            f"init-image-strength:{self.init_image_strength} "
            f"arch:{self.model_architecture.aliases[0]} "
            f"weights:{self.model_weights.aliases[0]}"
        )

    def logging_dict(self):
        """Return a dict of the object but with binary data replaced with reprs."""
        data = self.model_dump()
        data["init_image"] = repr(self.init_image)
        data["mask_image"] = repr(self.mask_image)
        data["image_prompt"] = repr(self.image_prompt)
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

    def make_concrete_copy(self) -> Self:
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
        result_images=None,
        performance_stats=None,
        progress_latents=None,
    ):
        import torch

        from imaginairy.utils import get_device, get_hardware_description
        from imaginairy.utils.img_utils import (
            model_latent_to_pillow_img,
            torch_img_to_pillow_img,
        )

        self.prompt = prompt

        self.images = {"generated": img}

        if result_images:
            for img_type, r_img in result_images.items():
                if r_img is None:
                    continue
                if isinstance(r_img, torch.Tensor):
                    if r_img.shape[1] == 4:
                        r_img = model_latent_to_pillow_img(r_img)
                    else:
                        r_img = torch_img_to_pillow_img(r_img)
                self.images[img_type] = r_img

        self.performance_stats = performance_stats
        self.progress_latents = progress_latents

        # for backward compat
        self.img = img

        self.is_nsfw = is_nsfw
        self.safety_score = safety_score
        self.created_at = datetime.now(tz=timezone.utc)
        self.torch_backend = get_device()
        self.hardware_name = get_hardware_description(get_device())

    def md5(self) -> str:
        return hashlib.md5(self.img.tobytes()).hexdigest()

    def metadata_dict(self):
        return {
            "prompt": self.prompt.logging_dict(),
        }

    def timings_str(self) -> str:
        if not self.performance_stats:
            return ""
        return " ".join(
            f"{k}:{v['duration']:.2f}s" for k, v in self.performance_stats.items()
        )

    def total_time(self) -> float:
        if not self.performance_stats:
            return 0
        return self.performance_stats["total"]["duration"]

    def gpu_str(self, stat_name="memory_peak") -> str:
        if not self.performance_stats:
            return ""
        return " ".join(
            f"{k}:{v[stat_name]/(10**6):.1f}MB"
            for k, v in self.performance_stats.items()
        )

    def _exif(self) -> "Image.Exif":
        from PIL import Image

        exif = Image.Exif()
        exif[ExifCodes.ImageDescription] = self.prompt.prompt_description()
        exif[ExifCodes.UserComment] = json.dumps(self.metadata_dict())
        # help future web scrapes not ingest AI generated art
        sd_version = self.prompt.model_weights.name
        if len(sd_version) > 40:
            sd_version = "custom weights"
        exif[ExifCodes.Software] = f"Imaginairy / Stable Diffusion {sd_version}"
        exif[ExifCodes.DateTime] = self.created_at.isoformat(sep=" ")[:19]
        exif[ExifCodes.HostComputer] = f"{self.torch_backend}:{self.hardware_name}"
        return exif

    def save(self, save_path: "Path | str", image_type: str = "generated") -> None:
        img = self.images.get(image_type, None)
        if img is None:
            msg = f"Image of type {image_type} not stored. Options are: {self.images.keys()}"
            raise ValueError(msg)

        img.convert("RGB").save(save_path, exif=self._exif())


class SafetyMode(str, Enum):
    STRICT = "strict"
    RELAXED = "relaxed"

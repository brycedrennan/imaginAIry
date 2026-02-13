"""Classes for image generation API models"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Extra, Field, HttpUrl, validator

from imaginairy.http_app.utils import Base64Bytes
from imaginairy.schema import ImaginePrompt


class StableStudioPrompt(BaseModel):
    text: str | None = None
    weight: float | None = Field(None, ge=-1, le=1)


class StableStudioModel(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    image: HttpUrl | None = None


class StableStudioStyle(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    image: HttpUrl | None = None


class StableStudioSolver(BaseModel):
    id: str
    name: str | None = None


class StableStudioInputImage(BaseModel):
    blob: Base64Bytes | None = None
    weight: float | None = Field(None, ge=0, le=1)


class StableStudioImage(BaseModel):
    id: str
    created_at: datetime | None = None
    input: Optional["StableStudioInput"] = None
    blob: Base64Bytes | None = None


class StableStudioImages(BaseModel):
    id: str
    exclusive_start_image_id: str | None = None
    images: list[StableStudioImage] | None = None


class StableStudioInput(BaseModel, extra=Extra.forbid):
    prompts: list[StableStudioPrompt] | None = None
    model: str | None = None
    style: str | None = None
    width: int | None = None
    height: int | None = None
    solver: StableStudioSolver | None = Field(None, alias="sampler")
    cfg_scale: float | None = Field(None, alias="cfgScale")
    steps: int | None = None
    seed: int | None = None
    mask_image: StableStudioInputImage | None = Field(None, alias="maskImage")
    initial_image: StableStudioInputImage | None = Field(None, alias="initialImage")

    @validator("seed")
    def validate_seed(cls, v):
        if v == 0:
            return None
        return v

    def to_imagine_prompt(self):
        """Converts this StableStudioInput to an ImaginePrompt."""
        from io import BytesIO

        from PIL import Image

        positive_prompt = self.prompts[0].text if self.prompts else None
        if self.prompts and len(self.prompts) > 1:
            negative_prompt = self.prompts[1].text if len(self.prompts) > 1 else None
        else:
            negative_prompt = None

        init_image = None
        init_image_strength = None
        if self.initial_image:
            init_image = self.initial_image.blob
            init_image_strength = self.initial_image.weight

        mask_image = self.mask_image.blob if self.mask_image else None

        solver_type = self.solver.id if self.solver else None

        return ImaginePrompt(
            prompt=positive_prompt,
            prompt_strength=self.cfg_scale,
            negative_prompt=negative_prompt,
            model_weights=self.model,
            solver_type=solver_type,
            seed=self.seed,
            steps=self.steps,
            size=(self.width, self.height),
            init_image=Image.open(BytesIO(init_image)) if init_image else None,
            init_image_strength=init_image_strength,
            mask_image=Image.open(BytesIO(mask_image)) if mask_image else None,
            mask_mode="keep",
        )


class StableStudioBatchRequest(BaseModel):
    input: StableStudioInput
    count: int = 1


class StableStudioBatchResponse(BaseModel):
    images: list[StableStudioImage]


StableStudioInput.model_rebuild()
StableStudioImage.model_rebuild()

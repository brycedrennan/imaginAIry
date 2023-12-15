"""Classes for image generation API models"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Extra, Field, HttpUrl, validator

from imaginairy.http_app.utils import Base64Bytes
from imaginairy.schema import ImaginePrompt


class StableStudioPrompt(BaseModel):
    text: Optional[str] = None
    weight: Optional[float] = Field(None, ge=-1, le=1)


class StableStudioModel(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    image: Optional[HttpUrl] = None


class StableStudioStyle(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    image: Optional[HttpUrl] = None


class StableStudioSolver(BaseModel):
    id: str
    name: Optional[str] = None


class StableStudioInputImage(BaseModel):
    blob: Optional[Base64Bytes] = None
    weight: Optional[float] = Field(None, ge=0, le=1)


class StableStudioImage(BaseModel):
    id: str
    created_at: Optional[datetime] = None
    input: Optional["StableStudioInput"] = None
    blob: Optional[Base64Bytes] = None


class StableStudioImages(BaseModel):
    id: str
    exclusive_start_image_id: Optional[str] = None
    images: Optional[List[StableStudioImage]] = None


class StableStudioInput(BaseModel, extra=Extra.forbid):
    prompts: Optional[List[StableStudioPrompt]] = None
    model: Optional[str] = None
    style: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    solver: Optional[StableStudioSolver] = Field(None, alias="sampler")
    cfg_scale: Optional[float] = Field(None, alias="cfgScale")
    steps: Optional[int] = None
    seed: Optional[int] = None
    mask_image: Optional[StableStudioInputImage] = Field(None, alias="maskImage")
    initial_image: Optional[StableStudioInputImage] = Field(None, alias="initialImage")

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
    images: List[StableStudioImage]


StableStudioInput.model_rebuild()
StableStudioImage.model_rebuild()

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Extra, Field, HttpUrl, validator

from imaginairy.http.utils import Base64Bytes


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


class StableStudioSampler(BaseModel):
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
    sampler: Optional[StableStudioSampler] = None
    cfg_scale: Optional[float] = Field(None, alias="cfgScale")
    steps: Optional[int] = None
    seed: Optional[int] = None
    mask_image: Optional[StableStudioInputImage] = Field(None, alias="maskImage")
    initial_image: Optional[StableStudioInputImage] = Field(None, alias="initialImage")

    @validator("seed")
    def validate_seed(cls, v):  # noqa
        if v == 0:
            return None
        return v


class StableStudioBatchRequest(BaseModel):
    input: StableStudioInput
    count: int = 1


class StableStudioBatchResponse(BaseModel):
    images: List[StableStudioImage]


StableStudioInput.update_forward_refs()
StableStudioImage.update_forward_refs()

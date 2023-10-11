import uuid

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from imaginairy.http_app.stablestudio.models import (
    StableStudioBatchRequest,
    StableStudioBatchResponse,
    StableStudioImage,
    StableStudioModel,
    StableStudioSampler,
)
from imaginairy.http_app.utils import generate_image_b64

router = APIRouter()


@router.post("/generate", response_model=StableStudioBatchResponse)
async def generate(studio_request: StableStudioBatchRequest):
    from imaginairy.http_app.app import gpu_lock

    generated_images = []
    imagine_prompt = studio_request.input.to_imagine_prompt()
    starting_seed = imagine_prompt.seed if imagine_prompt.seed is not None else None

    for run_num in range(studio_request.count):
        if starting_seed is not None:
            imagine_prompt.seed = starting_seed + run_num
        async with gpu_lock:
            img_base64 = await run_in_threadpool(generate_image_b64, imagine_prompt)

        image = StableStudioImage(id=str(uuid.uuid4()), blob=img_base64)
        generated_images.append(image)

    return StableStudioBatchResponse(images=generated_images)


@router.get("/samplers")
async def list_samplers():
    from imaginairy.config import SAMPLER_TYPE_OPTIONS

    sampler_objs = []
    for sampler_type in SAMPLER_TYPE_OPTIONS:
        sampler_obj = StableStudioSampler(id=sampler_type, name=sampler_type)
        sampler_objs.append(sampler_obj)

    return sampler_objs


@router.get("/models")
async def list_models():
    from imaginairy.config import MODEL_CONFIGS

    model_objs = []
    for model_config in MODEL_CONFIGS:
        if "inpaint" in model_config.description.lower():
            continue
        model_obj = StableStudioModel(
            id=model_config.short_name,
            name=model_config.description,
            description=model_config.description,
        )
        model_objs.append(model_obj)

    return model_objs

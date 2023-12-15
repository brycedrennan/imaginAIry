"""API routes for image generation service"""

import uuid

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from imaginairy.http_app.stablestudio.models import (
    StableStudioBatchRequest,
    StableStudioBatchResponse,
    StableStudioImage,
    StableStudioModel,
    StableStudioSolver,
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
    from imaginairy.config import SOLVER_CONFIGS

    sampler_objs = []

    for solver_config in SOLVER_CONFIGS:
        sampler_obj = StableStudioSolver(
            id=solver_config.aliases[0], name=solver_config.aliases[0]
        )
        sampler_objs.append(sampler_obj)

    return sampler_objs


@router.get("/models")
async def list_models():
    from imaginairy.config import MODEL_WEIGHT_CONFIGS

    model_objs = []
    for model_config in MODEL_WEIGHT_CONFIGS:
        if "inpaint" in model_config.name.lower():
            continue
        if model_config.architecture.output_modality != "image":
            continue
        model_obj = StableStudioModel(
            id=model_config.aliases[0],
            name=model_config.name,
            description=model_config.name,
        )
        model_objs.append(model_obj)

    return model_objs

from asyncio import Lock
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel  # noqa

from imaginairy import ImaginePrompt, imagine
from imaginairy.log_utils import configure_logging

app = FastAPI()

lock = Lock()


class ImagineWebPrompt(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    prompt: Optional[str]
    negative_prompt: Optional[str]
    prompt_strength: float = 7.5
    # init_image: Optional[Union[LazyLoadingImage, str]]
    init_image_strength: Optional[float] = None
    # control_inputs: Optional[List[ControlInput]] = None
    mask_prompt: Optional[str] = None
    # mask_image: Optional[Union[LazyLoadingImage, str]] = None
    mask_mode: str = "replace"
    mask_modify_original: bool = True
    outpaint: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    upscale: bool = False
    fix_faces: bool = False
    fix_faces_fidelity: float = 0.2
    # sampler_type: str = Field(..., alias='config.DEFAULT_SAMPLER')  # update the alias based on actual config field name
    conditioning: Optional[str] = None
    tile_mode: str = ""
    allow_compose_phase: bool = True
    # model: str = Field(..., alias='config.DEFAULT_MODEL')  # update the alias based on actual config field name
    model_config_path: Optional[str] = None
    is_intermediate: bool = False
    collect_progress_latents: bool = False
    caption_text: str = ""


def generate_image(prompt: ImagineWebPrompt):
    prompt = ImaginePrompt(prompt.prompt)
    result = next(imagine([prompt]))
    return result.images["generated"]


@app.post("/api/imagine")
async def imagine_endpoint(prompt: ImagineWebPrompt):
    async with lock:
        img = await run_in_threadpool(generate_image, prompt)
        img_io = BytesIO()
        img.save(img_io, "JPEG")
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/jpg")


@app.get("/api/imagine")
async def imagine_get_endpoint(text: str = Query(...)):
    async with lock:
        img = await run_in_threadpool(generate_image, ImagineWebPrompt(prompt=text))
        img_io = BytesIO()
        img.save(img_io, "JPEG")
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/jpg")


if __name__ == "__main__":
    import uvicorn

    configure_logging()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

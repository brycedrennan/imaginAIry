import logging
import os.path
import sys
import traceback
from asyncio import Lock

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from imaginairy.http.models import ImagineWebPrompt
from imaginairy.http.stablestudio import routes
from imaginairy.http.utils import generate_image

logger = logging.getLogger(__name__)

gpu_lock = Lock()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/stablestudio")


@app.post("/api/imagine")
async def imagine_endpoint(prompt: ImagineWebPrompt):
    async with gpu_lock:
        img_io = await run_in_threadpool(generate_image, prompt)
        return StreamingResponse(img_io, media_type="image/jpg")


@app.get("/api/imagine")
async def imagine_get_endpoint(text: str = Query(...)):
    async with gpu_lock:
        img_io = await run_in_threadpool(generate_image, ImagineWebPrompt(prompt=text))
        return StreamingResponse(img_io, media_type="image/jpg")


static_folder = os.path.dirname(os.path.abspath(__file__)) + "/stablestudio/dist"
print(f"static_folder: {static_folder}")

app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {exc}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )

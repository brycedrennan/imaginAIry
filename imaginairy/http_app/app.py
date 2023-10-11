import logging
import os.path
import sys
import traceback
from asyncio import Lock

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from imaginairy.http_app.stablestudio import routes
from imaginairy.http_app.utils import generate_image
from imaginairy.schema import ImaginePrompt

logger = logging.getLogger(__name__)

static_folder = os.path.dirname(os.path.abspath(__file__)) + "/stablestudio/dist"


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
async def imagine_endpoint(prompt: ImaginePrompt):
    async with gpu_lock:
        img_io = await run_in_threadpool(generate_image, prompt)
        return StreamingResponse(img_io, media_type="image/jpg")


@app.get("/api/imagine")
async def imagine_get_endpoint(text: str = Query(...)):
    async with gpu_lock:
        img_io = await run_in_threadpool(generate_image, ImaginePrompt(prompt=text))
        return StreamingResponse(img_io, media_type="image/jpg")


@app.get("/edit")
async def edit_redir():
    return FileResponse(f"{static_folder}/index.html")


@app.get("/generate")
async def generate_redir():
    return FileResponse(f"{static_folder}/index.html")


app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {exc}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )

import _thread
import logging
import os
import shutil
import time
from functools import lru_cache
from queue import Queue
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from imaginairy.utils import get_device
from imaginairy.utils.downloads import get_cached_url_path

from .msssim import ssim_matlab
from .RIFE_HDv3 import Model

logger = logging.getLogger(__name__)


def transfer_audio(sourceVideo, targetVideo):
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:
        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a copy -vn {tempAudioFileName}')

    targetNoAudio = (
        os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    )
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system(
        f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"'
    )

    if (
        os.path.getsize(targetVideo) == 0
    ):  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system(
            f'ffmpeg -y -i "{sourceVideo}" -c:a aac -b:a 160k -vn {tempAudioFileName}'
        )
        os.system(
            f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"'
        )
        if (
            os.path.getsize(targetVideo) == 0
        ):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print(
                "Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead."
            )

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")


RIFE_WEIGHTS_URL = "https://huggingface.co/imaginairy/rife-interpolation/resolve/26442e52cc30b88c5cb490702647b8de9aaee8a7/rife-flownet-4.13.2.safetensors"


@lru_cache(maxsize=1)
def load_rife_model(model_path=None, version=4.13, device=None):
    if model_path is None:
        model_path = RIFE_WEIGHTS_URL
    model_path = get_cached_url_path(model_path)
    device = device if device else get_device()
    model = Model()
    model.load_model(model_path, version=version)
    model.eval()
    model.flownet.to(device)
    return model


def make_inference(I0, I1, n, *, model, scale):
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n // 2, model=model, scale=scale)
        second_half = make_inference(middle, I1, n=n // 2, model=model, scale=scale)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


def interpolate_video_file(
    video_path: str | None = None,
    images_source_path: str | None = None,
    scale: float = 1.0,
    vid_out_name: str | None = None,
    target_fps: float | None = None,
    fps_multiplier: int = 2,
    model_weights_path: str | None = None,
    fp16: bool = False,
    montage: bool = False,
    png_out: bool = False,
    output_extension: str = "mp4",
    device=None,
):
    assert video_path is not None or images_source_path is not None
    assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    device = device if device else get_device()

    if images_source_path is not None:
        png_out = True

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  # type: ignore

    model = load_rife_model(model_weights_path, version=4.13)
    logger.info(f"Loaded RIFE from {model_weights_path}")

    if video_path is not None:
        import skvideo.io

        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if target_fps is None:
            fpsNotAssigned = True
            target_fps = fps * fps_multiplier
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(video_path)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # type: ignore
        video_path_wo_ext, ext = os.path.splitext(video_path)
        print(
            f"{video_path_wo_ext}.{output_extension}, {tot_frame} frames in total, {fps}FPS to {target_fps}FPS"
        )
        if png_out is False and fpsNotAssigned is True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
    else:
        assert images_source_path is not None
        videogen = []
        for f in os.listdir(images_source_path):
            if "png" in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key=lambda x: int(x[:-4]))
        lastframe = cv2.imread(
            os.path.join(images_source_path, videogen[0]), cv2.IMREAD_UNCHANGED
        )[:, :, ::-1].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape

    vid_out = None
    if png_out:
        if not os.path.exists("vid_out"):
            os.mkdir("vid_out")
    else:
        if vid_out_name is None:
            assert video_path_wo_ext is not None
            assert target_fps is not None
            vid_out_name = f"{video_path_wo_ext}_{fps_multiplier}X_{int(np.round(target_fps))}fps.{output_extension}"
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, target_fps, (w, h))  # type: ignore

    def clear_write_buffer(png_out, write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if png_out:
                cv2.imwrite(f"vid_out/{cnt:0>7d}.png", item[:, :, ::-1])
                cnt += 1
            else:
                vid_out.write(item[:, :, ::-1])

    def build_read_buffer(img, montage, read_buffer, videogen):
        try:
            for frame in videogen:
                if img is not None:
                    frame = cv2.imread(os.path.join(img, frame), cv2.IMREAD_UNCHANGED)[
                        :, :, ::-1
                    ].copy()
                if montage:
                    frame = frame[:, left : left + w]
                read_buffer.put(frame)
        except Exception as e:  # noqa
            print(f"skipping frame due to error: {e}")
        read_buffer.put(None)

    def pad_image(img):
        if fp16:
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    if montage:
        left = w // 4
        w = w // 2
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    if montage:
        lastframe = lastframe[:, left : left + w]
    write_buffer: Queue = Queue(maxsize=500)
    read_buffer: Queue = Queue(maxsize=500)
    _thread.start_new_thread(
        build_read_buffer, (images_source_path, montage, read_buffer, videogen)
    )
    _thread.start_new_thread(clear_write_buffer, (png_out, write_buffer))

    I1 = (
        torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    I1 = pad_image(I1)
    temp = None  # save lastframe when processing static frame

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = (
            torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get()  # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = (
                torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                .to(device, non_blocking=True)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < 0.2:
            output = []
            for i in range(fps_multiplier - 1):
                output.append(I0)
            """
            output = []
            step = 1 / fps_multiplier
            alpha = 0
            for i in range(fps_multiplier - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            """
        else:
            output = make_inference(
                I0, I1, fps_multiplier - 1, model=model, scale=scale
            )

        if montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)  # type: ignore
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)  # type: ignore
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    if montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)

    while not write_buffer.empty():
        time.sleep(0.1)
    pbar.close()
    if vid_out is not None:
        vid_out.release()
    assert vid_out_name is not None

    # move audio to new video file if appropriate
    if png_out is False and fpsNotAssigned is True and video_path is not None:
        try:
            transfer_audio(video_path, vid_out_name)
        except Exception as e:  # noqa
            logger.info(
                f"Audio transfer failed. Interpolated video will have no audio. {e}"
            )
            targetNoAudio = (
                os.path.splitext(vid_out_name)[0]
                + "_noaudio"
                + os.path.splitext(vid_out_name)[1]
            )
            os.rename(targetNoAudio, vid_out_name)


def pad_image(img, scale):
    tmp = max(128, int(128 / scale))
    ph, pw = (
        ((img.shape[1] - 1) // tmp + 1) * tmp,
        ((img.shape[2] - 1) // tmp + 1) * tmp,
    )
    padding = (0, pw - img.shape[2], 0, ph - img.shape[1])
    return F.pad(img, padding)


def interpolate_images(
    image_list,
    scale=1.0,
    fps_multiplier=2,
    model_weights_path=None,
    device=None,
) -> List[Image.Image]:
    assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    torch.set_grad_enabled(False)
    device = device if device else get_device()
    model = load_rife_model(model_weights_path, version=4.13)

    interpolated_images = []
    for i in range(len(image_list) - 1):
        I0 = image_to_tensor(image_list[i], device)
        I1 = image_to_tensor(image_list[i + 1], device)
        # I0, I1 = pad_image(I0, scale), pad_image(I1, scale)

        interpolated = make_inference(
            I0, I1, n=fps_multiplier - 1, model=model, scale=scale
        )
        interpolated_images.append(image_list[i])
        for img in interpolated:
            img = (img[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            interpolated_images.append(Image.fromarray(img))

    interpolated_images.append(image_list[-1])
    return interpolated_images


def image_to_tensor(image, device):
    """
    Converts a PIL image to a PyTorch tensor.

    Args:
    - image (PIL.Image): The image to convert.
    - device (torch.device): The device to use (CPU or CUDA).

    Returns:
    - torch.Tensor: The image converted to a PyTorch tensor.
    """
    tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1)))
    tensor = tensor.to(device, non_blocking=True).unsqueeze(0).float() / 255.0
    return tensor

import os.path

import cv2
import torch

from imaginairy.img_utils import (
    add_caption_to_image,
    imgpaths_to_imgs,
    model_latents_to_pillow_imgs,
    pillow_img_to_opencv_img,
)
from imaginairy.utils import shrink_list


def make_bounce_animation(
    imgs,
    outpath,
    transition_duration_ms=500,
    start_pause_duration_ms=1000,
    end_pause_duration_ms=2000,
):
    first_img = imgs[0]
    last_img = imgs[-1]
    middle_imgs = imgs[1:-1]
    max_fps = 20
    max_frames = int(round(transition_duration_ms / 1000 * max_fps))
    min_duration = int(1000 / 20)
    if middle_imgs:
        progress_duration = int(round(transition_duration_ms / len(middle_imgs)))
    else:
        progress_duration = 0
    progress_duration = max(progress_duration, min_duration)

    middle_imgs = shrink_list(middle_imgs, max_frames)

    frames = [first_img, *middle_imgs, last_img, *list(reversed(middle_imgs))]

    # convert from latents
    converted_frames = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = model_latents_to_pillow_imgs(frame)[0]
        converted_frames.append(frame)
    frames = converted_frames

    durations = (
        [start_pause_duration_ms]
        + [progress_duration] * len(middle_imgs)
        + [end_pause_duration_ms]
        + [progress_duration] * len(middle_imgs)
    )

    make_animation(imgs=frames, outpath=outpath, frame_duration_ms=durations)


def make_slideshow_animation(
    imgs,
    outpath,
    image_pause_ms=1000,
):
    # convert from latents
    converted_frames = []
    for frame in imgs:
        if isinstance(frame, torch.Tensor):
            frame = model_latents_to_pillow_imgs(frame)[0]
        converted_frames.append(frame)

    durations = [image_pause_ms] * len(converted_frames)

    make_animation(imgs=converted_frames, outpath=outpath, frame_duration_ms=durations)


def make_animation(imgs, outpath, frame_duration_ms=100, captions=None):
    imgs = imgpaths_to_imgs(imgs)
    ext = os.path.splitext(outpath)[1].lower().strip(".")

    if captions:
        if len(captions) != len(imgs):
            raise ValueError("Captions and images must be of same length.")
        for img, caption in zip(imgs, captions):
            add_caption_to_image(img, caption)

    if ext == "gif":
        make_gif_animation(
            imgs=imgs, outpath=outpath, frame_duration_ms=frame_duration_ms
        )
    elif ext == "mp4":
        make_mp4_animation(
            imgs=imgs, outpath=outpath, frame_duration_ms=frame_duration_ms
        )


def make_gif_animation(imgs, outpath, frame_duration_ms=100, loop=0):
    imgs = imgpaths_to_imgs(imgs)
    imgs[0].save(
        outpath,
        save_all=True,
        append_images=imgs[1:],
        duration=frame_duration_ms,
        loop=loop,
        optimize=False,
    )


def make_mp4_animation(imgs, outpath, frame_duration_ms=50, fps=30, codec="mp4v"):
    imgs = imgpaths_to_imgs(imgs)
    frame_size = imgs[0].size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(outpath, fourcc, fps, frame_size)
    if not isinstance(frame_duration_ms, list):
        frame_duration_ms = [frame_duration_ms] * len(imgs)
    try:
        for image in select_images_by_duration_at_fps(imgs, frame_duration_ms, fps):
            image = pillow_img_to_opencv_img(image)
            out.write(image)
    finally:
        out.release()


def select_images_by_duration_at_fps(images, durations_ms, fps=30):
    """select the proper image to show for each frame of a video."""
    for i, image in enumerate(images):
        duration = durations_ms[i] / 1000
        num_frames = int(round(duration * fps))
        print(
            f"Showing image {i} for {num_frames} frames for {durations_ms[i]}ms at {fps} fps."
        )
        for j in range(num_frames):
            yield image

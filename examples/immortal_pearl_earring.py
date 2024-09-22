import os

import cv2
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

from imaginairy.api import imagine
from imaginairy.schema import ImaginePrompt, LazyLoadingImage, WeightedPrompt
from imaginairy.utils.log_utils import configure_logging


def generate_image_morph_video():
    base_image = LazyLoadingImage(
        filepath="tests/data/girl_with_a_pearl_earring_large.jpg"
    )
    output_dir = "outputs/video/pearl_earring"

    os.makedirs(output_dir, exist_ok=True)
    prompts = []
    seed = 290124740

    for year in frange(1900, 2100, 1 / 6):
        pearl_weight = max(2000 - year, 1)
        robotic_weight = min(max(year - 2025, 0), 25) * 2
        print(
            f"year: {year}, pearl_weight: {pearl_weight}, robotic_weight: {robotic_weight}"
        )
        scene = "scenic majestic mountains in the background"
        color_txt = "color" if year >= 1960 else "black and white"
        year_txt = str(int(year)) if year <= 2025 else "futuristic"
        transcendence_weight = (max(year - 2050, 0) / 3) * robotic_weight
        subprompts = [
            WeightedPrompt(
                text=f"{year_txt} professional {color_txt} headshot photo of a woman with a pearl earring  wearing an {year_txt} outfit. {scene}",
                weight=pearl_weight + 0.1,
            ),
            WeightedPrompt(
                text="photo of a cybernetic woman computer chips in her head. circuits, cybernetic, robotic, biomechanical, elegant, sharp focus, highly detailed, intricate details. scenic majestic mountains of mars in the background",
                weight=robotic_weight + 0.01,
            ),
            WeightedPrompt(
                text="photo of a cybernetic woman floating above a wormhole. computer chips in her head. circuits, cybernetic, robotic, biomechanical, elegant, sharp focus, highly detailed, intricate details",
                weight=transcendence_weight,
            ),
        ]
        prompt = ImaginePrompt(
            subprompts,
            model="SD-2.0-depth",
            negative_prompt="b&w, grayscale, ugly, old, deformed, disfigured",
            width=768,
            height=768,
            init_image=base_image,
            init_image_strength=0.0,
            seed=seed,
            sampler_type="plms",
            steps=30,
        )
        title = f"{year:.0f}"
        prompts.append((title, prompt))

    for i, (title, prompt) in tqdm(enumerate(prompts), total=len(prompts)):
        filename = os.path.join(output_dir, f"{i:04d}_{title}.jpg")

        if os.path.exists(filename):
            continue

        result = next(iter(imagine([prompt])))
        generated_image = result.images["generated"]

        draw = ImageDraw.Draw(generated_image)

        font_size = 48
        font = ImageFont.truetype("scripts/Roboto-Medium.ttf", font_size)

        x = 10
        y = 10

        draw.text((x, y), title, font=font, fill=(255, 255, 255))

        result.save(filename)

    create_video(
        output_dir,
        "outputs/video/pearl_earring.mp4",
        fps=60,
        frame_size=(result.img.width, result.img.height),
        codec="MP4V",
    )


def frange(start, stop, step=1.0):
    value = start
    while value < stop:
        yield value
        value += step


def create_video(image_dir, output_file, fps=60, frame_size=(640, 480), codec="MP4V"):
    images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
    images.sort()
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    for image in images:
        img = cv2.imread(os.path.join(image_dir, image))
        out.write(img)

    out.release()


if __name__ == "__main__":
    configure_logging()
    generate_image_morph_video()

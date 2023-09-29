import logging

from PIL import Image, ImageEnhance, ImageStat

from imaginairy import ImaginePrompt, imagine
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.schema import ControlNetInput

logger = logging.getLogger(__name__)


def colorize_img(img, max_width=1024, max_height=1024, caption=None):
    if not caption:
        caption = generate_caption(img, min_length=10)
        caption = caption.replace("black and white", "color")
        caption = caption.replace("old picture", "professional color photo")
        caption = caption.replace("vintage photograph", "professional color photo")
        caption = caption.replace("old photo", "professional color photo")
        caption = caption.replace("vintage photo", "professional color photo")
        caption = caption.replace("old color", "color")
        caption = caption.replace(" old fashioned ", " ")
        caption = caption.replace(" old time ", " ")
        caption = caption.replace(" old ", " ")
        logger.info(caption)
    control_inputs = [
        ControlNetInput(mode="colorize", image=img, strength=2),
    ]
    prompt_add = ". color photo, sharp-focus, highly detailed, intricate, Canon 5D"
    prompt = ImaginePrompt(
        prompt=f"{caption}{prompt_add}",
        init_image=img,
        init_image_strength=0.0,
        control_inputs=control_inputs,
        width=min(img.width, max_width),
        height=min(img.height, max_height),
        steps=30,
        prompt_strength=12,
    )
    result = next(iter(imagine(prompt)))
    colorized_img = replace_color(img, result.images["generated"])

    # allows the algorithm some leeway for the overall brightness of the image
    # results look better with this
    colorized_img = match_brightness(colorized_img, result.images["generated"])

    return colorized_img


def replace_color(target_img, color_src_img):
    color_src_img = color_src_img.resize(target_img.size)

    _, _, value = target_img.convert("HSV").split()
    hue, saturation, _ = color_src_img.convert("HSV").split()

    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")


def calculate_brightness(image):
    greyscale_image = image.convert("L")
    stat = ImageStat.Stat(greyscale_image)
    return stat.mean[0]


def match_brightness(target_img, source_img):
    target_brightness = calculate_brightness(target_img)
    source_brightness = calculate_brightness(source_img)

    brightness_factor = source_brightness / target_brightness

    enhancer = ImageEnhance.Brightness(target_img)
    return enhancer.enhance(brightness_factor)

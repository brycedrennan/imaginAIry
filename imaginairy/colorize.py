from PIL import Image

from imaginairy import ImaginePrompt, imagine
from imaginairy.enhancers.describe_image_blip import generate_caption


def colorize_img(img):
    caption = generate_caption(img)
    caption = caption.replace("black and white", "color")

    prompt = ImaginePrompt(
        prompt=caption,
        init_image=img,
        init_image_strength=0.01,
        control_image=img,
        control_mode="hed",
        negative_prompt="black and white",
        # width=img.width,
        # height=img.height,
    )
    result = list(imagine(prompt))[0]
    colorized_img = replace_color(img, result.images["generated"])

    prompt = ImaginePrompt(
        prompt=caption,
        init_image=colorized_img,
        init_image_strength=0.1,
        control_image=img,
        control_mode="hed",
        negative_prompt="black and white",
        width=min(img.width, 1024),
        height=min(img.height, 1024),
        steps=30,
    )
    result = list(imagine(prompt))[0]
    colorized_img = replace_color(img, result.images["generated"])
    return colorized_img


def replace_color(target_img, color_src_img):
    color_src_img = color_src_img.resize(target_img.size)

    _, _, value = target_img.convert("HSV").split()
    hue, saturation, _ = color_src_img.convert("HSV").split()

    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")

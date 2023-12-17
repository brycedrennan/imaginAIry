from typing import Literal

import pyparsing
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageColor import getrgb

from imaginairy.utils.img_utils import create_halo_effect
from imaginairy.utils.paths import PKG_ROOT
from imaginairy.utils.spaced_kv_parser import parse_spaced_key_value_pairs


def determine_max_font_size(
    text: str,
    draw: ImageDraw.ImageDraw,
    font_path: str,
    width: int,
    height: int,
    margin_pct: float,
    line_spacing: int = 4,
) -> int:
    """
    Determine the maximum font size that allows the text to fit within the given image dimensions and margin constraints.
    Updated to use multiline_textbbox in Pillow 10.1.0.

    :param text: Text to be drawn.
    :param draw: ImageDraw object to measure text size.
    :param font_path: Path to the font file.
    :param width: Width of the image.
    :param height: Height of the image.
    :param margin_pct: Margin percentage.
    :return: Maximum font size.
    """
    max_width = width - 2 * (width * margin_pct)
    max_height = height - 2 * (height * margin_pct)

    font_size = 1
    font = ImageFont.truetype(font_path, font_size)

    while True:
        # Use multiline_textbbox to get the bounding box of the text
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
        text_width = bbox[2] - bbox[0]  # right - left
        text_height = bbox[3] - bbox[1]  # bottom - top

        if text_width > max_width or text_height > max_height:
            break
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)

    # Subtract 1 because the loop exits after the size becomes too large
    return font_size - 1


def generate_word_image(
    text: str,
    width: int,
    height: int,
    margin_pct: float = 0.1,
    line_spacing: int = 4,
    text_align: Literal["left", "center", "right"] = "center",
    font_path: str = f"{PKG_ROOT}/data/DejaVuSans.ttf",
    font_color: str = "black",
    background_color: str = "white",
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)

    max_font_size = determine_max_font_size(
        text, draw, font_path, width, height, margin_pct, line_spacing=line_spacing
    )

    font = ImageFont.truetype(font_path, max_font_size)

    bbox = draw.multiline_textbbox((0, 0), text, font=font)

    # Calculate text position
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) / 2
    y = (height - text_height) / 2 - bbox[1]

    draw.multiline_text(
        (x, y), text, fill=font_color, font=font, align=text_align, spacing=line_spacing
    )

    return image


def image_from_textimg_str(text: str, width: int, height: int) -> Image.Image:
    """
    Create an image from a textimg string.
    """
    try:
        data = parse_spaced_key_value_pairs(text)
    except pyparsing.ParseException:
        raise ValueError("Invalid format for textimg")  # noqa

    first_key = next(iter(data))

    if first_key != "textimg":
        raise ValueError("Invalid format for textimg")

    allowed_keys = {
        "textimg",
        "font",
        "font_color",
        "background_color",
        "text_align",
        "line_spacing",
        "margin_pct",
        "halo",
    }
    submitted_keys = set(data.keys())
    invalid_keys = submitted_keys - allowed_keys
    if invalid_keys:
        msg = f"Invalid attributes for textimg: '{invalid_keys}'. Valid attributes are '{allowed_keys}'"
        raise ValueError(msg)

    text_align = data.get("text_align", "center")
    valid_alignments = {"left", "center", "right"}
    if text_align not in valid_alignments:
        msg = f"Invalid text_align '{text_align}'. Valid options are 'left', 'center' and 'right'"
        raise ValueError(msg)
    assert text_align in valid_alignments
    background_color: str = data.get("background_color", "white")
    img = generate_word_image(
        text=data["textimg"].replace("\\n", "\n"),
        width=width,
        height=height,
        margin_pct=float(data.get("margin_pct", 0.1)),
        line_spacing=int(data.get("line_spacing", 4)),
        text_align=text_align,  # type: ignore
        font_path=data.get("font", f"{PKG_ROOT}/data/DejaVuSans.ttf"),
        font_color=data.get("font_color", "black"),
        background_color=background_color,
    )
    bg_color_rgb = getrgb(background_color)
    if data.get("halo", "0").lower() in ("true", "1", "yes"):
        img = create_halo_effect(img, background_color=bg_color_rgb)

    return img

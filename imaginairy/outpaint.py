import re

from PIL import Image, ImageDraw


def prepare_image_for_outpaint(
    img, mask=None, up=None, down=None, left=None, right=None, _all=0, snap_multiple=8
):
    up = up if up is not None else _all
    down = down if down is not None else _all
    left = left if left is not None else _all
    right = right if right is not None else _all

    lft_pct = left / (left + right)
    rgt_pct = right / (left + right)
    up_pct = up / (up + down)
    dwn_pct = down / (up + down)

    new_width = round((img.width + left + right) / snap_multiple) * snap_multiple
    new_height = round((img.height + up + down) / snap_multiple) * snap_multiple
    height_addition = max(new_height - img.height, 0)
    width_addition = max(new_width - img.width, 0)
    up = int(round(height_addition * up_pct))
    down = int(round(height_addition * dwn_pct))
    left = int(round(width_addition * lft_pct))
    right = int(round(width_addition * rgt_pct))

    expanded_image = Image.new(
        "RGB", (img.width + left + right, img.height + up + down), (0, 0, 0)
    )
    expanded_image.paste(img, (left, up))

    # extend border pixels outward, this helps prevents lines at the boundary because masks getting reduced to
    # 64x64 latent space can cause some inaccuracies

    if up > 0:
        expanded_image.paste(
            img.crop((0, 0, img.width, 1)).resize((expanded_image.width, up)),
            (0, 0),
        )
        expanded_image.paste(
            img.crop((0, 0, img.width, 1)).resize((img.width, up)),
            (left, 0),
        )
    if down > 0:
        expanded_image.paste(
            img.crop((0, img.height - 1, img.width, img.height)).resize(
                (expanded_image.width, down)
            ),
            (0, expanded_image.height - down),
        )
        expanded_image.paste(
            img.crop((0, img.height - 1, img.width, img.height)).resize(
                (img.width, down)
            ),
            (left, expanded_image.height - down),
        )
    if left > 0:
        expanded_image.paste(
            img.crop((0, 0, 1, img.height)).resize((left, expanded_image.height)),
            (0, 0),
        )
        expanded_image.paste(
            img.crop((0, 0, 1, img.height)).resize((left, img.height)),
            (0, up),
        )
    if right > 0:
        expanded_image.paste(
            img.crop((img.width - 1, 0, img.width, img.height)).resize(
                (right, expanded_image.height)
            ),
            (expanded_image.width - right, 0),
        )
        expanded_image.paste(
            img.crop((img.width - 1, 0, img.width, img.height)).resize(
                (right, img.height)
            ),
            (expanded_image.width - right, up),
        )

    # create a mask for the new boundaries
    expanded_mask = Image.new("L", (expanded_image.width, expanded_image.height), 255)
    if mask is None:
        draw = ImageDraw.Draw(expanded_mask)
        draw.rectangle(
            (left, up, left + img.width, up + img.height), fill="black", outline="black"
        )
    else:
        expanded_mask.paste(mask, (left, up))

    return expanded_image, expanded_mask


def outpaint_arg_str_parse(arg_str):
    arg_pattern = re.compile(r"([A-Z]+)(\d+)")

    args = arg_str.upper().split(",")
    valid_directions = ["up", "down", "left", "right", "all"]
    valid_direction_chars = {c[0]: c for c in valid_directions}
    kwargs = {}
    for arg in args:
        match = arg_pattern.match(arg)
        if not match:
            raise ValueError(f"Invalid outpaint argument '{arg}'")
        direction, amount = match.groups()
        direction = direction.lower()
        if len(direction) == 1:
            if direction not in valid_direction_chars:
                raise ValueError(f"Invalid outpaint direction '{direction}'")
            direction = valid_direction_chars[direction]
        elif direction not in valid_directions:
            raise ValueError(f"Invalid outpaint direction '{direction}'")
        kwargs[direction] = int(amount)

    if "all" in kwargs:
        kwargs["_all"] = kwargs.pop("all")

    return kwargs

import re

import torch
from PIL import Image, ImageDraw
from torch import nn

from imaginairy.img_utils import torch_img_to_pillow_img


def outpaint_calculations(
    img_width,
    img_height,
    up=None,
    down=None,
    left=None,
    right=None,
    _all=0,
    snap_multiple=8,
):
    up = up if up is not None else _all
    down = down if down is not None else _all
    left = left if left is not None else _all
    right = right if right is not None else _all

    lft_pct = left / (left + right) if left + right else 0
    rgt_pct = right / (left + right) if left + right else 0
    up_pct = up / (up + down) if up + down else 0
    dwn_pct = down / (up + down) if up + down else 0

    new_width = round((img_width + left + right) / snap_multiple) * snap_multiple
    new_height = round((img_height + up + down) / snap_multiple) * snap_multiple
    height_addition = max(new_height - img_height, 0)
    width_addition = max(new_width - img_width, 0)
    up = int(round(height_addition * up_pct))
    down = int(round(height_addition * dwn_pct))
    left = int(round(width_addition * lft_pct))
    right = int(round(width_addition * rgt_pct))

    return up, down, left, right, new_width, new_height


def prepare_tensor_for_outpaint(
    img, mask=None, up=None, down=None, left=None, right=None, _all=0, snap_multiple=8
):
    up, down, left, right, new_width, new_height = outpaint_calculations(
        img_width=img.shape[2],
        img_height=img.shape[1],
        up=up,
        down=down,
        left=left,
        right=right,
        _all=_all,
        snap_multiple=snap_multiple,
    )

    def resize(img_t, h, w):
        new_size = (img_t.shape[0], h, w)
        return nn.functional.interpolate(img_t, size=new_size, mode="nearest")

    def paste(dst, src, y, x):
        dst[:, y : y + src.shape[1], x : x + src.shape[2]] = src

    expanded_img = torch.zeros(
        img.shape[0], img.shape[1] + up + down, img.shape[2] + left + right
    )
    expanded_img[:, up : up + img.shape[1], left : left + img.shape[2]] = img

    # extend border pixels outward, this helps prevents lines at the boundary because masks getting reduced to
    # 64x64 latent space can cause some inaccuracies

    if up > 0:
        top_row = img[:, 0, :]
        paste(expanded_img, resize(top_row, h=up, w=expanded_img.shape[2]), y=0, x=0)
        paste(expanded_img, resize(top_row, h=up, w=img.shape[2]), y=0, x=left)

    if down > 0:
        bottom_row = img[:, -1, :]
        paste(
            expanded_img,
            resize(bottom_row, h=down, w=expanded_img.shape[2]),
            y=expanded_img.shape[1] - down,
            x=0,
        )
        paste(
            expanded_img,
            resize(bottom_row, h=down, w=img.shape[2]),
            y=expanded_img.shape[1] - down,
            x=left,
        )

    if left > 0:
        left_column = img[:, :, 0]
        paste(
            expanded_img, resize(left_column, h=expanded_img.shape[1], w=left), y=0, x=0
        )
        paste(expanded_img, resize(left_column, h=img.shape[1], w=left), y=up, x=0)

    if right > 0:
        right_column = img[:, :, -1]
        paste(
            expanded_img,
            resize(right_column, h=expanded_img.shape[1], w=right),
            y=0,
            x=expanded_img.shape[2] - right,
        )
        paste(
            expanded_img,
            resize(right_column, h=img.shape[1], w=right),
            y=up,
            x=expanded_img.shape[2] - right,
        )

    # create a mask for the new boundaries
    expanded_mask = torch.zeros_like(expanded_img)

    if mask is None:
        # set to black
        expanded_mask[:, up : up + img.shape[1], left : left + img.shape[2]] = 1
    else:
        expanded_mask[:, up : up + mask.shape[1], left : left + mask.shape[2]] = mask

    return expanded_img, expanded_mask


def prepare_image_for_outpaint(
    img, mask=None, up=None, down=None, left=None, right=None, _all=0, snap_multiple=8
):
    up, down, left, right, new_width, new_height = outpaint_calculations(
        img_width=img.width,
        img_height=img.height,
        up=up,
        down=down,
        left=left,
        right=right,
        _all=_all,
        snap_multiple=snap_multiple,
    )
    ran_img_t = torch.randn((1, 3, new_height, new_width), device="cpu")
    expanded_image = torch_img_to_pillow_img(ran_img_t)
    # expanded_image = Image.new(
    #     "RGB", (img.width + left + right, img.height + up + down), (0, 0, 0)
    # )
    expanded_image.paste(img, (left, up))

    # extend border pixels outward, this helps prevents lines at the boundary because masks getting reduced to
    # 64x64 latent space can cause some inaccuracies
    alpha = 20
    if up > 0:
        top_row = img.crop((0, 0, img.width, 1))
        top_row.putalpha(alpha)
        expanded_image.paste(
            top_row.resize((expanded_image.width, up)),
            (0, 0),
        )
        expanded_image.paste(
            top_row.resize((img.width, up)),
            (left, 0),
        )
    if down > 0:
        bottom_row = img.crop((0, img.height - 1, img.width, img.height))
        bottom_row.putalpha(alpha)
        expanded_image.paste(
            bottom_row.resize((expanded_image.width, down)),
            (0, expanded_image.height - down),
        )
        expanded_image.paste(
            bottom_row.resize((img.width, down)),
            (left, expanded_image.height - down),
        )
    if left > 0:
        left_column = img.crop((0, 0, 1, img.height))
        left_column.putalpha(alpha)
        expanded_image.paste(
            left_column.resize((left, expanded_image.height)),
            (0, 0),
        )
        expanded_image.paste(
            left_column.resize((left, img.height)),
            (0, up),
        )
    if right > 0:
        right_column = img.crop((img.width - 1, 0, img.width, img.height))
        right_column.putalpha(alpha)
        expanded_image.paste(
            right_column.resize((right, expanded_image.height)),
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
    if not arg_str:
        return {}
    arg_pattern = re.compile(r"([A-Z]+)(\d+)")

    args = arg_str.upper().split(",")
    valid_directions = ["up", "down", "left", "right", "all"]
    valid_direction_chars = {c[0]: c for c in valid_directions}
    kwargs = {}
    for arg in args:
        match = arg_pattern.match(arg)
        if not match:
            msg = f"Invalid outpaint argument '{arg}'"
            raise ValueError(msg)
        direction, amount = match.groups()
        direction = direction.lower()
        if len(direction) == 1:
            if direction not in valid_direction_chars:
                msg = f"Invalid outpaint direction '{direction}'"
                raise ValueError(msg)
            direction = valid_direction_chars[direction]
        elif direction not in valid_directions:
            msg = f"Invalid outpaint direction '{direction}'"
            raise ValueError(msg)
        kwargs[direction] = int(amount)

    if "all" in kwargs:
        kwargs["_all"] = kwargs.pop("all")

    return kwargs

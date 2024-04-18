import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor

logger = logging.getLogger(__name__)


def tile_process(
    img: "Tensor",
    scale: int,
    model: "torch.nn.Module",
    tile_size: int = 512,
    tile_pad: int = 50,
) -> "Tensor":
    """
    Process an image by tiling it, processing each tile, and then merging them back into one image.

    Args:
    img (Tensor): The input image tensor.
    scale (int): The scale factor for the image.
    tile_size (int): The size of each tile.
    tile_pad (int): The padding for each tile.
    model (torch.nn.Module): The model used for processing the tile.

    Returns:
    Tensor: The processed output image.
    """
    import torch

    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # Initialize the output tensor
    output = img.new_zeros(output_shape)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    logger.debug(f"Tiling with {tiles_x}x{tiles_y} ({tiles_x*tiles_y}) tiles")

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate the input tile coordinates with and without padding
            ofs_x, ofs_y = x * tile_size, y * tile_size
            input_start_x, input_end_x = ofs_x, min(ofs_x + tile_size, width)
            input_start_y, input_end_y = ofs_y, min(ofs_y + tile_size, height)
            padded_start_x, padded_end_x = (
                max(input_start_x - tile_pad, 0),
                min(input_end_x + tile_pad, width),
            )
            padded_start_y, padded_end_y = (
                max(input_start_y - tile_pad, 0),
                min(input_end_y + tile_pad, height),
            )

            # Extract the input tile
            input_tile = img[
                :, :, padded_start_y:padded_end_y, padded_start_x:padded_end_x
            ]

            # Process the tile
            with torch.no_grad():
                output_tile = model(input_tile)

            # Calculate the output tile coordinates
            output_start_x, output_end_x = input_start_x * scale, input_end_x * scale
            output_start_y, output_end_y = input_start_y * scale, input_end_y * scale
            tile_output_start_x = (input_start_x - padded_start_x) * scale
            tile_output_end_x = (
                tile_output_start_x + (input_end_x - input_start_x) * scale
            )
            tile_output_start_y = (input_start_y - padded_start_y) * scale
            tile_output_end_y = (
                tile_output_start_y + (input_end_y - input_start_y) * scale
            )

            # Place the processed tile in the output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = (
                output_tile[
                    :,
                    :,
                    tile_output_start_y:tile_output_end_y,
                    tile_output_start_x:tile_output_end_x,
                ]
            )

    return output

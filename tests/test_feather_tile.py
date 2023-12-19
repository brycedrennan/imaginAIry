import itertools

import pytest

from imaginairy.schema import LazyLoadingImage
from imaginairy.utils.feather_tile import rebuild_image, tile_image, tile_setup
from imaginairy.utils.img_utils import (
    pillow_img_to_torch_image,
    torch_img_to_pillow_img,
)
from tests import TESTS_FOLDER

img_ratios = [0.2, 0.242, 0.3, 0.33333333, 0.5, 0.75, 1, 4 / 3.0, 16 / 9.0, 2, 21 / 9.0]
pcts = [
    0,
    0.09,
    0.1,
    0.2,
    0.25,
    0.3,
    1 / 3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
]
initial_sizes = [512]
flip = [True, False]

cases = [
    (1, 256, 0),
    (1, 256, 0.125),
    (1, 256, 0.25),
    (1, 256, 0.5),
    (1, 128, 0),
    (1, 128, 0.125),
    (1, 128, 0.25),
    (1, 128, 0.5),
    (1, 512, 0),
    (0.2, 46, 0.09),
    (0.2, 46, 0.1),
    (0.242, 46, 0.2),
    (0.2, 51, 1 / 3.0),
    (0.2, 102, 0.09),  # tile size same as width of image
]


@pytest.mark.parametrize(("img_ratio", "tile_size", "overlap_pct"), cases)
def test_feather_tile_simple(img_ratio, tile_size, overlap_pct):
    img = pillow_img_to_torch_image(
        LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bowl_of_fruit.jpg")
    )
    img = img[:, :, : img.shape[2], : int(img.shape[3] * img_ratio)]
    img_sum = img.sum()
    tiles = tile_image(img, tile_size=tile_size, overlap_percent=overlap_pct)
    tile_coords, tile_size, overlap = tile_setup(
        tile_size, overlap_pct, (img.size(2), img.size(3))
    )

    # print(
    #     f"tile_coords={tile_coords}, tile_size={tile_size}, overlap={overlap}, img.shape={img.shape}"
    # )

    rebuilt = rebuild_image(
        tiles, base_img=img, tile_size=tile_size, overlap_percent=overlap_pct
    )
    assert rebuilt.shape == img.shape
    diff = abs(float(rebuilt.sum()) - float(img_sum))
    if diff >= 1:
        torch_img_to_pillow_img(img).show()
        torch_img_to_pillow_img(rebuilt).show()
        torch_img_to_pillow_img(rebuilt - img).show()

    assert diff < 1


@pytest.mark.skip(
    reason="takes too long. runs 5000 scenarios. if you mess with feather_tile, run this test"
)
def test_feather_tile_brute():
    source_img = pillow_img_to_torch_image(
        LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bowl_of_fruit.jpg")
    )

    def tile_untile(img, tile_size, overlap_percent):
        img_sum = img.sum()
        tiles = tile_image(img, tile_size=tile_size, overlap_percent=overlap_percent)
        tile_coords, tile_size, overlap = tile_setup(
            tile_size, overlap_percent, (img.size(2), img.size(3))
        )
        # print(
        #     f"tile_coords={tile_coords}, tile_size={tile_size}, overlap={overlap}, img.shape={img.shape}"
        # )

        rebuilt = rebuild_image(
            tiles, base_img=img, tile_size=tile_size, overlap_percent=overlap_percent
        )
        assert rebuilt.shape == img.shape
        diff = abs(float(rebuilt.sum()) - float(img_sum))
        if diff > 1:
            torch_img_to_pillow_img(img).show()
            torch_img_to_pillow_img(rebuilt).show()
            torch_img_to_pillow_img((rebuilt - img) * 20).show()

        else:
            pass
        # print(
        #     f"{status}: img:{img.shape} tile_size={tile_size} overlap_percent={overlap_percent} diff={diff}"
        # )
        assert diff < 1

    for tile_size_pct, overlap_percent, img_ratio, flip_ratio in itertools.product(
        pcts, pcts, img_ratios, flip
    ):
        if flip_ratio:
            img = source_img.clone()[:, :, :, : int(source_img.shape[3] * img_ratio)]
        else:
            img = source_img.clone()[:, :, : int(source_img.shape[2] * img_ratio), :]
        tile_size = int(source_img.shape[3] * tile_size_pct)
        if not tile_size:
            continue

        if overlap_percent >= 0.5:
            continue

        # print(
        #     f"img_ratio={img_ratio}, tile_size_pct={tile_size_pct}, overlap_percent={overlap_percent}, tile_size={tile_size} img.shape={img.shape}"
        # )
        tile_untile(img, tile_size=tile_size, overlap_percent=overlap_percent)
        del img

    # tile_untile(img, tile_size=256, overlap_percent=0.25)

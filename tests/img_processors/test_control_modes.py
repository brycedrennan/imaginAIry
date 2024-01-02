import itertools

import pytest

from imaginairy.img_processors.control_modes import CONTROL_MODES, create_depth_map
from imaginairy.modules.midas.api import ISL_PATHS
from imaginairy.schema import LazyLoadingImage
from imaginairy.utils import seed_everything
from imaginairy.utils.img_utils import (
    pillow_img_to_torch_image,
    torch_img_to_pillow_img,
)
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


def control_img_to_pillow_img(img_t):
    return torch_img_to_pillow_img((img_t - 0.5) * 2)


control_mode_params = list(CONTROL_MODES.items())


@pytest.mark.skip()
def test_compare_depth_maps(filename_base_for_outputs):
    sizes = [384, 512, 768]
    model_types = ISL_PATHS
    img = LazyLoadingImage(
        url="https://zhyever.github.io/patchfusion/images/interactive/case6.png"
    )
    for model_type, size in itertools.product(model_types.keys(), sizes):
        if (
            "dpt_swin" in model_type
            or "next_vit" in model_type
            or "levit" in model_type
        ):
            continue

        print(f"Testing {model_type} with size {size}")

        img_t = pillow_img_to_torch_image(img)

        depth_t = create_depth_map(img_t, model_type=model_type, max_size=size)
        depth_img = control_img_to_pillow_img(depth_t)
        img_path = f"{filename_base_for_outputs}_{model_type}_{size}.png"
        depth_img.save(img_path)


@pytest.mark.parametrize(("control_name", "control_func"), control_mode_params)
def test_control_images(filename_base_for_outputs, control_func, control_name):
    seed_everything(42)
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2.png")
    img_t = pillow_img_to_torch_image(img)
    if control_name == "inpaint":
        control_t = control_func(img_t.clone(), img_t.clone())
    else:
        control_t = control_func(img_t.clone())
    control_img = control_img_to_pillow_img(control_t)
    img_path = f"{filename_base_for_outputs}.png"

    assert_image_similar_to_expectation(control_img, img_path, threshold=8000)

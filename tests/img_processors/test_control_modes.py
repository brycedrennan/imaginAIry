import pytest

from imaginairy import LazyLoadingImage
from imaginairy.img_processors.control_modes import CONTROL_MODES
from imaginairy.img_utils import pillow_img_to_torch_image, torch_img_to_pillow_img
from tests import TESTS_FOLDER
from tests.utils import assert_image_similar_to_expectation


def control_img_to_pillow_img(img_t):
    return torch_img_to_pillow_img((img_t - 0.5) * 2)


control_mode_params = list(CONTROL_MODES.items())


@pytest.mark.parametrize("control_name,control_func", control_mode_params)
def test_control_images(filename_base_for_outputs, control_func, control_name):
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2.png")
    img_t = pillow_img_to_torch_image(img)

    control_t = control_func(img_t.clone())
    control_img = control_img_to_pillow_img(control_t)
    img_path = f"{filename_base_for_outputs}.png"

    assert_image_similar_to_expectation(control_img, img_path, threshold=4000)

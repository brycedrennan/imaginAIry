import pytest
from pydantic import ValidationError

from imaginairy.schema import ControlInput, LazyLoadingImage
from tests import TESTS_FOLDER


@pytest.fixture(name="lazy_img")
def _lazy_img():
    return LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png")


def test_controlnetinput_basic(lazy_img):
    ControlInput(mode="canny", image=lazy_img)
    ControlInput(mode="canny", image_raw=lazy_img)


def test_controlnetinput_invalid_mode(lazy_img):
    with pytest.raises(ValueError, match=r".*Invalid controlnet mode.*"):
        ControlInput(mode="pizza", image=lazy_img)


def test_controlnetinput_both_images(lazy_img):
    with pytest.raises(ValueError, match=r".*cannot specify both.*"):
        ControlInput(mode="canny", image=lazy_img, image_raw=lazy_img)


def test_controlnetinput_filepath_input(lazy_img):
    """Test that we accept filepaths here."""
    c = ControlInput(mode="canny", image=f"{TESTS_FOLDER}/data/red.png")
    c.image.convert("RGB")
    c = ControlInput(mode="canny", image_raw=f"{TESTS_FOLDER}/data/red.png")
    c.image_raw.convert("RGB")


def test_controlnetinput_big(lazy_img):
    ControlInput(mode="canny", strength=2)
    with pytest.raises(ValidationError, match=r".*float_type.*"):
        ControlInput(mode="canny", strength=2**2048)

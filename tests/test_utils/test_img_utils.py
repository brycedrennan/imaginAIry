from imaginairy.utils.img_utils import create_halo_effect
from imaginairy.utils.text_image import generate_word_image
from tests import TESTS_FOLDER


def test_create_halo_effect():
    """Test if the image has the correct dimensions"""
    width, height = 1920, 1080
    bg_shade = 245
    img = generate_word_image("OBEY", width, height, font_color="black")
    img.save(f"{TESTS_FOLDER}/data/obey.png")

    img = create_halo_effect(img, (bg_shade, bg_shade, bg_shade))
    img.save(f"{TESTS_FOLDER}/data/obey-halo.png")

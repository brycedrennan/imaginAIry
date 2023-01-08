import logging

from imaginairy import LazyLoadingImage
from imaginairy.enhancers.facecrop import generate_face_crops
from tests import TESTS_FOLDER

logger = logging.getLogger(__name__)


def test_facecrop():
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2.png")
    crops = generate_face_crops(img, max_width=img.width, max_height=img.height)

import logging

from imaginairy.enhancers.facecrop import generate_face_crops
from imaginairy.schema import LazyLoadingImage
from tests import TESTS_FOLDER

logger = logging.getLogger(__name__)


def test_facecrop():
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/bench2.png")
    generate_face_crops((50, 50, 150, 150), max_width=img.width, max_height=img.height)

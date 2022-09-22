import hashlib

import pytest
from PIL import Image
from pytorch_lightning import seed_everything

from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.enhancers.describe_image_clip import find_img_text_similarity
from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


def test_fix_faces():
    img = Image.open(f"{TESTS_FOLDER}/data/distorted_face.png")
    seed_everything(1)
    img = enhance_faces(img)
    img.save(f"{TESTS_FOLDER}/test_output/fixed_face.png")
    if "mps" in get_device():
        assert img_hash(img) == "a75991307eda675a26eeb7073f828e93"
    else:
        assert img_hash(img) == "e56c1205bbc8f251be05773f2ba7fa24"


def img_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def test_clip_masking():
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    pred = get_img_mask(img, "head")
    pred.save(f"{TESTS_FOLDER}/test_output/earring_mask.png")


def test_describe_picture():
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    caption = generate_caption(img)
    assert caption == "a painting of a girl with a pearl ear"


def test_clip_text_comparison():
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring.jpg")
    phrases = [
        "Johannes Vermeer painting",
        "a painting of a girl with a pearl earring",
        "a bulldozer",
        "photo",
    ]
    probs = find_img_text_similarity(img, phrases)
    assert probs[:2] == [
        (
            "a painting of a girl with a pearl earring",
            pytest.approx(0.2857227921485901, rel=1e-3),
        ),
        ("Johannes Vermeer painting", pytest.approx(0.25186583399772644, rel=1e-3)),
    ]

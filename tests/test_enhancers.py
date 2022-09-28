import hashlib

import pytest
from PIL import Image
from pytorch_lightning import seed_everything

from imaginairy import ImaginePrompt, imagine
from imaginairy.enhancers.bool_masker import MASK_PROMPT
from imaginairy.enhancers.clip_masking import get_img_mask
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.enhancers.describe_image_clip import find_img_text_similarity
from imaginairy.enhancers.face_restoration_codeformer import enhance_faces
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


@pytest.mark.skipif(
    get_device() == "cpu", reason="TypeError: Got unsupported ScalarType BFloat16"
)
def test_fix_faces():
    img = Image.open(f"{TESTS_FOLDER}/data/distorted_face.png")
    seed_everything(1)
    img = enhance_faces(img)
    img.save(f"{TESTS_FOLDER}/test_output/fixed_face.png")
    if "mps" in get_device():
        assert img_hash(img) == "a75991307eda675a26eeb7073f828e93"
    else:
        # probably different based on whether first run or not. looks the same either way
        assert img_hash(img) in ["c840cf3bfe5a7760734f425a3f8941cf", "e56c1205bbc8f251be05773f2ba7fa24"]


def img_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_clip_masking():
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring_large.jpg")

    for mask_modifier in [
        "*0.5",
        "*1",
        "*6",
    ]:
        pred_bin, pred_grayscale = get_img_mask(
            img,
            f"face AND NOT (bandana OR hair OR blue fabric){{{mask_modifier}}}",
            threshold=0.5,
        )
        pred_grayscale.save(
            f"{TESTS_FOLDER}/test_output/earring_mask_{mask_modifier}_g.png"
        )
        pred_bin.save(
            f"{TESTS_FOLDER}/test_output/earring_mask_{mask_modifier}_bin.png"
        )

    prompt = ImaginePrompt(
        "a female firefighter in front of a burning building",
        init_image=img,
        init_image_strength=0.95,
        # lower steps for faster tests
        steps=40,
        mask_prompt="(head OR face){*5}",
        mask_mode="replace",
        upscale=False,
        fix_faces=True,
    )

    result = next(imagine(prompt))
    result.save(
        f"{TESTS_FOLDER}/test_output/earring_mask_photo.png",
        image_type="modified_original",
    )


boolean_mask_test_cases = [
    (
        "fruit bowl",
        "'fruit bowl'",
    ),
    (
        "((((fruit bowl))))",
        "'fruit bowl'",
    ),
    (
        "fruit OR bowl",
        "('fruit' OR 'bowl')",
    ),
    (
        "fruit|bowl",
        "('fruit' OR 'bowl')",
    ),
    (
        "fruit | bowl",
        "('fruit' OR 'bowl')",
    ),
    (
        "fruit OR bowl OR pear",
        "('fruit' OR 'bowl' OR 'pear')",
    ),
    (
        "fruit AND bowl",
        "('fruit' AND 'bowl')",
    ),
    (
        "fruit & bowl",
        "('fruit' AND 'bowl')",
    ),
    (
        "fruit AND NOT green",
        "('fruit' AND NOT 'green')",
    ),
    (
        "fruit bowl{+0.5}",
        "'fruit bowl'+0.5",
    ),
    (
        "fruit bowl{+0.5} OR fruit",
        "('fruit bowl'+0.5 OR 'fruit')",
    ),
    (
        "NOT pizza",
        "NOT 'pizza'",
    ),
    (
        "car AND (wheels OR trunk OR engine OR windows) AND NOT (truck OR headlights{*10})",
        "('car' AND ('wheels' OR 'trunk' OR 'engine' OR 'windows') AND NOT ('truck' OR 'headlights'*10))",
    ),
    (
        "car AND (wheels OR trunk OR engine OR windows OR headlights) AND NOT (truck OR headlights){*10}",
        "('car' AND ('wheels' OR 'trunk' OR 'engine' OR 'windows' OR 'headlights') AND NOT ('truck' OR 'headlights')*10)",
    ),
]


@pytest.mark.parametrize("mask_text,expected", boolean_mask_test_cases)
def test_clip_mask_parser(mask_text, expected):
    parsed = MASK_PROMPT.parseString(mask_text)[0][0]
    assert str(parsed) == expected


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
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
            pytest.approx(0.2857227921485901, abs=0.01),
        ),
        ("Johannes Vermeer painting", pytest.approx(0.25186583399772644, abs=0.01)),
    ]

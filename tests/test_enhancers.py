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
from tests.utils import assert_image_similar_to_expectation


@pytest.mark.skipif(
    get_device() == "cpu", reason="TypeError: Got unsupported ScalarType BFloat16"
)
def test_fix_faces(filename_base_for_orig_outputs, filename_base_for_outputs):
    distorted_img = Image.open(f"{TESTS_FOLDER}/data/distorted_face.png")
    seed_everything(1)
    img = enhance_faces(distorted_img)

    distorted_img.save(f"{filename_base_for_orig_outputs}__orig.jpg")
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(img, img_path=img_path, threshold=2800)


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_clip_masking(filename_base_for_outputs):
    img = Image.open(f"{TESTS_FOLDER}/data/girl_with_a_pearl_earring_large.jpg")

    for mask_modifier in ["*0.5", "*6", "+1", "+11", "+101", "-25"]:
        pred_bin, pred_grayscale = get_img_mask(
            img,
            f"face AND NOT (bandana OR hair OR blue fabric){{{mask_modifier}}}",
            threshold=0.5,
        )

        mask_modifier = mask_modifier.replace("*", "x")
        img_path = f"{filename_base_for_outputs}_mask{mask_modifier}_g.png"

        assert_image_similar_to_expectation(
            pred_grayscale, img_path=img_path, threshold=300
        )

        img_path = f"{filename_base_for_outputs}_mask{mask_modifier}_bin.png"
        assert_image_similar_to_expectation(pred_bin, img_path=img_path, threshold=10)

    prompt = ImaginePrompt(
        "",
        init_image=img,
        init_image_strength=0.5,
        # lower steps for faster tests
        steps=40,
        mask_prompt="(head OR face){*5}",
        mask_mode="keep",
        upscale=False,
        fix_faces=True,
        seed=42,
        sampler_type="plms",
    )

    result = next(imagine(prompt))
    img_path = f"{filename_base_for_outputs}.png"
    assert_image_similar_to_expectation(result.img, img_path=img_path, threshold=1200)


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
    assert caption in {
        "a painting of a girl with a pearl earring wearing a yellow dress and a pearl earring in her ear and a black background",
        "a painting of a girl with a pearl ear wearing a yellow dress and a pearl earring on her left ear and a black background",
    }


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
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

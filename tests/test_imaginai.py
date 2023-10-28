import pytest
from click.testing import CliRunner

from imaginairy.cli.imaginai import (
    convert_to_cli_command,
    format_keys,
    generate_cli_cmd,
    imaginai_cmd,
    improve_imagine_descriptions,
    list_option_descriptions,
)
from imaginairy.cli.imagine import imagine_cmd


def test_format_keys_simple():
    input_dict = {"key1": "value1", "key2": "value2"}
    expected_output = {"--key1": "value1", "--key2": "value2"}
    assert format_keys(input_dict) == expected_output


def test_with_colorize_prompt():
    parameters = {"--prompt_texts": "Colorize this image"}
    expected_output = "Colorize this image"
    assert convert_to_cli_command(parameters) == expected_output


def test_with_edit_prompt():
    parameters = {"--prompt_texts": "Edit this photo"}
    expected_output = "edit this photo "
    assert convert_to_cli_command(parameters) == expected_output


def test_with_upscale_prompt():
    parameters = {"--prompt_texts": "Upscale this picture"}
    expected_output = "upscale this picture"
    assert convert_to_cli_command(parameters) == expected_output


def test_with_generic_prompt():
    parameters = {"--prompt_texts": "This is a test"}
    expected_output = 'imagine "this is a test" '
    assert convert_to_cli_command(parameters) == expected_output


def test_with_additional_parameters():
    parameters = {"--prompt_texts": "This is a test", "--fix-faces": ""}
    expected_output = 'imagine "this is a test" --fix-faces '
    assert convert_to_cli_command(parameters) == expected_output


def test_with_mask_prompt_parameter():
    parameters = {"--prompt_texts": "This is a test", "--mask-prompt": ""}
    expected_output = 'imagine "this is a test" --mask-prompt  '
    assert convert_to_cli_command(parameters) == expected_output


def test_list_option_descriptions():
    assert len(list_option_descriptions(imagine_cmd)) > 30


def test_all_parameters_present():
    input_descriptions = {
        "--upscale": "old description 1",
        "--fix-faces": "old description 2",
        "--init-image": "old description 3",
        "--control-image": "old description 4",
    }
    expected_output = {
        "--upscale": "scale the image to be larger",
        "--fix-faces": "improves the quality of faces, fixes distorted faces caused by image generation",
        "--init-image": "the image to start with, if not specified, ImaginAIry will generate an image from scratch. Don't use this for anything that is related to the control image, use --control-image instead.",
        "--control-image": "image used for control signal in image generation, use this instead of --init-image if you want to use a control image",
    }
    assert improve_imagine_descriptions(input_descriptions) == expected_output


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_imaginai_cmd():
    runner = CliRunner()
    result = runner.invoke(imaginai_cmd, ["test prompt"])

    assert result.exit_code == 0
    assert "🎩 Chatgpt generating imaginAIry prompt." in result.output


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_openpose_cmd_generation():
    user_prompt = "Use the pose of the person in this image to make a new image with a polar bear in the same pose assets/indiana.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--control-image",
        "assets/indiana.jpg",
        "--control-mode",
        "openpose",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_canny_edge_cmd_generation():
    user_prompt = "I want a variation of this image of a woman using Canny Edge Control assets/lena.png."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--control-image",
        "assets/lena.png",
        "--control-mode",
        "canny",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_hed_boundary_control_cmd_generation():
    user_prompt = "Using HED Boundary Control, I want a variation of this image that is of a dalmatian dog.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "--control-image", "dog.jpg", "--control-mode", "hed"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_depth_map_control_cmd_generation():
    user_prompt = "Make some variations of this photo of my living room, using Depth Map Control fancy-living.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--control-image",
        "fancy-living.jpg",
        "--control-mode",
        "depth",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_normal_map_control_cmd_generation():
    user_prompt = (
        "Use Normal Map Control to create a variation of my photo of a bird bird.jpg."
    )

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "--control-image", "bird.jpg", "--control-mode", "normal"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_shuffle_control_cmd_generation():
    user_prompt = "Use Image Shuffle Control to generate a clown of this girl in my image pearl-girl.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--control-image",
        "pearl-girl.jpg",
        "--control-mode",
        "shuffle",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_editing_cmd_generation():
    user_prompt = "Make the an anime version of this girl in this image pearl-girl.jpg using Editing Instructions Control."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--control-image",
        "pearl-girl.jpg",
        "--control-mode",
        "edit",
        "--init-image-strength",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_details_cmd_generation():
    user_prompt = "Make this image have higher resolution assets/wishbone.jpg using  Add Details Control.."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--control-image",
        "assets/wishbone.jpg",
        "--control-mode",
        "details",
        "--init-image-strength",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_colorization_cmd_generation():
    user_prompt = "Make this black and white image have color pearl-girl.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["colorize", "pearl-girl.jpg"]
    assert len(key_words) == 2
    assert cli_command == "colorize pearl-girl.jpg"


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_edit_cmd_generation():
    user_prompt = (
        "Make this scene look winter scenic_landscape.jpg using InstructPix2Pix"
    )

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["edit", "-p", "scenic_landscape.jpg"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_masking_cmd_generation():
    user_prompt = "Take this photo, keep the face the same, but make her a firefighter pearl-girl.jpg using masking."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = [
        "imagine",
        "--init-image",
        "pearl-girl.jpg",
        "--mask-prompt",
        "--mask-mode",
        "--init-image-strength",
    ]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_fix_faces_cmd_generation():
    user_prompt = "a girl smiling, make sure the faces look good."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "--fix-faces"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_upscale_cmd_generation():
    user_prompt = "please make this image larger using upscaling my-image.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    assert cli_command == "upscale my-image.jpg"


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_tiled_image_cmd_generation():
    user_prompt = "make a tile of gold coins"

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "--tile", "gold coins"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_360_image_cmd_generation():
    user_prompt = "make a 360 image of desert landscape."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "-w", "-h", "--tile-x"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_image_to_image_cmd_generation():
    user_prompt = "I'd like to change the style of this portrait(a woman with pearl earrings), make it look professional, but the original image should remain mostly intact. Use image to image."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "--init-image", "--init-image-strength"]
    for word in key_words:
        assert word in cli_command


@pytest.mark.skip(reason="Makes external API calls to OpenAI")
def test_outpainting_cmd_generation():
    user_prompt = "I want to expand what this photo shows. A lot downward. A little bit to the sides, but not at all upward pearl-earring.jpg."

    cli_command = generate_cli_cmd(user_prompt=user_prompt)
    key_words = ["imagine", "--init-image", "--init-image-strength", "--outpaint"]
    for word in key_words:
        assert word in cli_command

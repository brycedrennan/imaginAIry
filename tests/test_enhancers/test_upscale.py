from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from imaginairy.cli.upscale import (
    upscale_cmd,
)
from tests import TESTS_FOLDER


@pytest.mark.parametrize(
    ("format_option", "outdir_option", "expected_directory", "expected_filename"),
    [
        # Test no given format with no outdir specified
        (None, None, "/tests/data/", "sand_upscale_difficult.upscaled.jpg"),
        # Test no given format with outdir specified
        (
            None,
            "tests/data/temp/",
            "tests/data/temp/",
            "sand_upscale_difficult.upscaled.jpg",
        ),
        # Test given format with no outdir specified
        (
            "{original_filename}{original_filename}.upscaled{file_extension}",
            None,
            "/tests/data/",
            "sand_upscale_difficultsand_upscale_difficult.upscaled.jpg",
        ),
        # Test given format and given directory
        (
            "{original_filename}{original_filename}.upscaled{file_extension}",
            "tests/data/temp/",
            "tests/data/temp/",
            "sand_upscale_difficultsand_upscale_difficult.upscaled.jpg",
        ),
        # Test default config with 'DEFAULT' keyword and no outdir specified
        ("DEFAULT", None, "/tests/data/", ".upscaled"),
        # Test 'DEV' config with no outdir specified
        (
            "DEV",
            None,
            "./outputs/upscaled",
            "000000_realesrgan-x2-plus_sand_upscale_difficult.upscaled.jpg",
        ),
        # Test 'DEFAULT' config with outdir specified
        (
            "DEFAULT",
            "tests/data/temp/",
            "tests/data/temp/",
            "tests/data/temp/sand_upscale_difficult.upscaled.jpg",
        ),
        # Test 'DEV' config with outdir specified
        (
            "DEV",
            "tests/data/temp/",
            "tests/data/temp/",
            "tests/data/temp/000000_realesrgan-x2-plus_sand_upscale_difficult.upscaled.jpg",
        ),
        # save directory specified in both format and outdir
        (
            "tests/data/temp/{original_filename}.upscaled{file_extension}",
            "tests/data/temp/",
            "tests/data/temp/",
            "tests/data/temp/sand_upscale_difficult.upscaled.jpg",
        ),
        # save directory specified in format but not outdir
        (
            "tests/data/temp/{original_filename}.upscaled{file_extension}",
            None,
            "/tests/data/temp/",
            "tests/data/temp/sand_upscale_difficult.upscaled.jpg",
        ),
    ],
)
def test_upscale_cmd_format_option(
    format_option, outdir_option, expected_directory, expected_filename
):
    runner = CliRunner()

    mock_img = Mock()
    mock_img.save = Mock()

    command_args = ["tests/data/sand_upscale_difficult.jpg"]
    if format_option:
        command_args.extend(["--format", format_option])
    if outdir_option:
        command_args.extend(["--outdir", outdir_option])

    with patch.multiple(
        "imaginairy.enhancers.upscale", upscale_image=Mock(return_value=mock_img)
    ), patch(
        "imaginairy.utils.glob_expand_paths",
        new=Mock(return_value=[f"{TESTS_FOLDER}/data/sand_upscale_difficult.jpg"]),
    ):
        result = runner.invoke(upscale_cmd, command_args)

        assert result.exit_code == 0
        assert "Saved to " in result.output
        mock_img.save.assert_called()  # Check if save method was called
        saved_path = mock_img.save.call_args[0][
            0
        ]  # Get the path where the image was saved

        assert expected_directory in saved_path
        assert expected_filename in saved_path

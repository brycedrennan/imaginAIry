from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from imaginairy.cli.upscale import (
    upscale_cmd,
)
from tests import TESTS_FOLDER


@pytest.fixture()
def mock_pil_save():
    with patch.object(Image, "save", autospec=True) as mock_save:
        yield mock_save


def test_upscale_cmd_format_option():
    runner = CliRunner()

    mock_img = Mock()
    mock_img.save = Mock()

    with patch.multiple(
        "imaginairy.enhancers.upscale", upscale_image=Mock(return_value=mock_img)
    ), patch(
        "imaginairy.utils.glob_expand_paths",
        new=Mock(return_value=[f"{TESTS_FOLDER}/data/sand_upscale_difficult.jpg"]),
    ):
        result = runner.invoke(
            upscale_cmd,
            [
                "tests/data/sand_upscale_difficult.jpg",
                "--format",
                "{original_filename}_upscaled_{file_sequence_number}_{algorithm}_{now}",
            ],
        )

        assert result.exit_code == 0
        assert "Saved to " in result.output
        mock_img.save.assert_called()  # Check if save method was called

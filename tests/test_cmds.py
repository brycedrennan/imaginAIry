import pytest
from click.testing import CliRunner

from imaginairy.cmds import edit_image, imagine_cmd
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_imagine_cmd():
    runner = CliRunner()
    result = runner.invoke(
        imagine_cmd,
        [
            "gold coins",
            "--steps",
            "25",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
            "--seed",
            "703425280",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_edit_cmd():
    runner = CliRunner()
    result = runner.invoke(
        edit_image,
        [
            f"{TESTS_FOLDER}/data/dog.jpg",
            "--steps",
            "1",
            "-p",
            "turn the dog into a cat",
        ],
    )
    assert result.exit_code == 0

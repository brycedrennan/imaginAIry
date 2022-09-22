import pytest
from click.testing import CliRunner

from imaginairy.cmds import imagine_cmd
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_imagine_cmd():
    runner = CliRunner()
    result = runner.invoke(
        imagine_cmd,
        ["gold coins", "--steps", "5", "--outdir", f"{TESTS_FOLDER}/test_output"],
    )
    assert result.exit_code == 0

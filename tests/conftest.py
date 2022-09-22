import sys

import pytest

from imaginairy import api
from imaginairy.suppress_logs import suppress_annoying_logs_and_warnings
from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    platform_appropriate_autocast,
)

if "pytest" in str(sys.argv):
    suppress_annoying_logs_and_warnings()


@pytest.fixture(scope="session", autouse=True)
def pre_setup():
    api.IMAGINAIRY_SAFETY_MODE = "disabled"
    suppress_annoying_logs_and_warnings()
    with fix_torch_nn_layer_norm(), fix_torch_group_norm(), platform_appropriate_autocast():
        yield

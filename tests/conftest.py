import logging
import os
import sys

import pytest
from urllib3 import HTTPConnectionPool

from imaginairy import api
from imaginairy.suppress_logs import suppress_annoying_logs_and_warnings
from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    platform_appropriate_autocast,
)
from tests import TESTS_FOLDER

if "pytest" in str(sys.argv):
    suppress_annoying_logs_and_warnings()

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def pre_setup():
    api.IMAGINAIRY_SAFETY_MODE = "disabled"
    suppress_annoying_logs_and_warnings()
    os.makedirs(f"{TESTS_FOLDER}/test_output", exist_ok=True)

    orig_urlopen = HTTPConnectionPool.urlopen

    def urlopen_tattle(self, method, url, *args, **kwargs):
        # traceback.print_stack()
        print(os.environ.get("PYTEST_CURRENT_TEST"))
        print(f"{method} {self.host}{url}")
        return orig_urlopen(self, method, url, *args, **kwargs)

    HTTPConnectionPool.urlopen = urlopen_tattle

    with fix_torch_nn_layer_norm(), fix_torch_group_norm(), platform_appropriate_autocast():
        yield

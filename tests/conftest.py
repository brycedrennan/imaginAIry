import logging
import os
import sys

import pytest
import responses
from urllib3 import HTTPConnectionPool

from imaginairy import api
from imaginairy.log_utils import suppress_annoying_logs_and_warnings
from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    get_device,
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
    # test_output_folder = f"{TESTS_FOLDER}/test_output"

    # delete the testoutput folder and recreate it
    # rmtree(test_output_folder)
    os.makedirs(f"{TESTS_FOLDER}/test_output", exist_ok=True)

    orig_urlopen = HTTPConnectionPool.urlopen

    def urlopen_tattle(self, method, url, *args, **kwargs):
        # traceback.print_stack()
        print(os.environ.get("PYTEST_CURRENT_TEST"))
        print(f"{method} {self.host}{url}")
        result = orig_urlopen(self, method, url, *args, **kwargs)
        print(f"{method} {self.host}{url} DONE")
        # raise HTTPError("NO NETWORK CALLS")
        return result

    HTTPConnectionPool.urlopen = urlopen_tattle

    with fix_torch_nn_layer_norm(), fix_torch_group_norm(), platform_appropriate_autocast():
        yield


@pytest.fixture(autouse=True)
def reset_get_device():
    get_device.cache_clear()


@pytest.fixture()
def filename_base_for_outputs(request):
    filename_base = f"{TESTS_FOLDER}/test_output/{request.node.name}_{get_device()}_"
    return filename_base


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        yield rsps

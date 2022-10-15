import logging
import os
import sys
from functools import partialmethod
from shutil import rmtree

import pytest
import responses
from tqdm import tqdm
from urllib3 import HTTPConnectionPool

from imaginairy import api
from imaginairy.log_utils import suppress_annoying_logs_and_warnings
from imaginairy.samplers.base import SAMPLER_TYPE_OPTIONS
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

SAMPLERS_FOR_TESTING = SAMPLER_TYPE_OPTIONS
if get_device() == "mps:0":
    SAMPLERS_FOR_TESTING = ["plms", "k_euler_a"]
elif get_device() == "cpu":
    SAMPLERS_FOR_TESTING = []


@pytest.fixture(scope="session", autouse=True)
def pre_setup():
    api.IMAGINAIRY_SAFETY_MODE = "disabled"
    suppress_annoying_logs_and_warnings()
    test_output_folder = f"{TESTS_FOLDER}/test_output"

    # delete the testoutput folder and recreate it
    try:
        rmtree(test_output_folder)
    except FileNotFoundError:
        pass
    os.makedirs(test_output_folder, exist_ok=True)

    orig_urlopen = HTTPConnectionPool.urlopen

    def urlopen_tattle(self, method, url, *args, **kwargs):
        # traceback.print_stack()
        # current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
        # print(f"{current_test} {method} {self.host}{url}")
        result = orig_urlopen(self, method, url, *args, **kwargs)

        # raise HTTPError("NO NETWORK CALLS")
        return result

    HTTPConnectionPool.urlopen = urlopen_tattle
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # real_randn = torch.randn
    # def randn_tattle(*args, **kwargs):
    #     print("RANDN CALL RANDN CALL")
    #     traceback.print_stack()
    #     return real_randn(*args, **kwargs)
    #
    # torch.randn = randn_tattle

    with fix_torch_nn_layer_norm(), fix_torch_group_norm(), platform_appropriate_autocast():
        yield


@pytest.fixture(autouse=True)
def reset_get_device():
    get_device.cache_clear()


@pytest.fixture()
def filename_base_for_outputs(request):
    filename_base = f"{TESTS_FOLDER}/test_output/{request.node.name}_"
    return filename_base


@pytest.fixture()
def filename_base_for_orig_outputs(request):
    filename_base = f"{TESTS_FOLDER}/test_output/{request.node.originalname}_"
    return filename_base


@pytest.fixture(params=SAMPLERS_FOR_TESTING)
def sampler_type(request):

    return request.param


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


def pytest_addoption(parser):
    parser.addoption(
        "--subset",
        action="store",
        default=None,
        help="Runs an exclusive subset of tests: '1/3', '2/3', '3/3'. Useful for distributed testing",
    )


@pytest.hookimpl()
def pytest_collection_modifyitems(config, items):
    """Only select a subset of tests to run, based on the --subset option."""
    filtered_node_ids = set()
    node_ids = [f.nodeid for f in items]
    node_ids.sort()
    subset = config.getoption("--subset")
    if subset:
        partition_no, total_partitions = subset.split("/")
        partition_no, total_partitions = int(partition_no), int(total_partitions)
        if partition_no < 1 or partition_no > total_partitions:
            raise ValueError("Invalid subset")
        for i, node_id in enumerate(node_ids):
            if i % total_partitions == partition_no - 1:
                filtered_node_ids.add(node_id)

        items[:] = [i for i in items if i.nodeid in filtered_node_ids]

        print(
            f"Running subset {partition_no}/{total_partitions} {len(filtered_node_ids)} tests:"
        )
        filtered_node_ids = list(filtered_node_ids)
        filtered_node_ids.sort()
        for n in filtered_node_ids:
            print(f"   {n}")

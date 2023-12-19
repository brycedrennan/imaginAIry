import contextlib
import csv
import gc
import logging
import os
import sys
from functools import partialmethod
from shutil import rmtree

import pytest
import responses
import torch.cuda
from tqdm import tqdm
from urllib3 import HTTPConnectionPool

from imaginairy import api
from imaginairy.api import imagine
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import (
    fix_torch_group_norm,
    fix_torch_nn_layer_norm,
    get_device,
    platform_appropriate_autocast,
)
from imaginairy.utils.log_utils import (
    configure_logging,
    suppress_annoying_logs_and_warnings,
)
from tests import TESTS_FOLDER

if "pytest" in str(sys.argv):
    suppress_annoying_logs_and_warnings()

logger = logging.getLogger(__name__)

# SOLVERS_FOR_TESTING = SOLVER_TYPE_OPTIONS
# if get_device() == "mps:0":
#     SOLVERS_FOR_TESTING = ["plms", "k_euler_a"]
# elif get_device() == "cpu":
#     SOLVERS_FOR_TESTING = []

SOLVERS_FOR_TESTING = ["ddim", "dpmpp"]


@pytest.fixture(scope="session", autouse=True)
def _pre_setup():
    api.IMAGINAIRY_SAFETY_MODE = "disabled"
    suppress_annoying_logs_and_warnings()
    test_output_folder = f"{TESTS_FOLDER}/test_output"

    # delete the testoutput folder and recreate it
    with contextlib.suppress(FileNotFoundError):
        rmtree(test_output_folder)
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
    configure_logging("DEBUG")

    with fix_torch_nn_layer_norm(), fix_torch_group_norm(), platform_appropriate_autocast():
        yield


@pytest.fixture(autouse=True)
def _reset_get_device():
    get_device.cache_clear()


@pytest.fixture()
def filename_base_for_outputs(request):
    filename_base = f"{TESTS_FOLDER}/test_output/{request.node.name}_"
    return filename_base


@pytest.fixture()
def filename_base_for_orig_outputs(request):
    filename_base = f"{TESTS_FOLDER}/test_output/{request.node.originalname}_"
    return filename_base


@pytest.fixture(params=SOLVERS_FOR_TESTING)
def solver_type(request):
    return request.param


@pytest.fixture()
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


@pytest.fixture(scope="session")
def default_model_loaded():
    """
    Just to make sure default weights are downloaded before the test runs

    """
    prompt = ImaginePrompt(
        "dogs lying on a hot pink couch",
        size=64,
        steps=2,
        seed=1,
        solver_type="ddim",
    )

    next(imagine(prompt))


cuda_tests_node_ids = []
cuda_test_tracker_filepath = f"{TESTS_FOLDER}/data/cuda-tests.csv"


@pytest.fixture(autouse=True)
def detect_cuda_tests(request):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.max_memory_allocated()
    yield
    if torch.cuda.is_available():
        end_memory = torch.cuda.max_memory_allocated()
        memory_diff = end_memory - start_memory
        if memory_diff > 0:
            test_id = request.node.nodeid
            print(f"Test {test_id} used {memory_diff} bytes of GPU memory")
            cuda_tests_node_ids.append(test_id)

            torch.cuda.empty_cache()
            gc.collect()


@pytest.hookimpl()
def pytest_collection_modifyitems(config, items):
    """Only select a subset of tests to run, based on the --subset option."""

    node_ids_to_mark = read_stored_cuda_test_nodes()
    for item in items:
        if item.nodeid in node_ids_to_mark:
            item.add_marker(pytest.mark.gputest)

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


def pytest_sessionstart(session):
    from imaginairy.utils.debug_info import get_debug_info

    debug_info = get_debug_info()

    for k, v in debug_info.items():
        if k == "nvidia_smi":
            continue
        k += ":"
        print(f"{k: <30} {v}")

    if "nvidia_smi" in debug_info:
        print(debug_info["nvidia_smi"])


def pytest_sessionfinish(session, exitstatus):
    existing_node_ids = read_stored_cuda_test_nodes()
    updated_node_ids = existing_node_ids.union(set(cuda_tests_node_ids))

    # Write updated, sorted list of node IDs to file
    with open(cuda_test_tracker_filepath, "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        for node_id in sorted(updated_node_ids):
            writer.writerow([node_id])


def read_stored_cuda_test_nodes():
    node_ids = set()
    try:
        with open(cuda_test_tracker_filepath, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                node_ids.add(row[0])
    except FileNotFoundError:
        pass  # File does not exist yet
    return node_ids

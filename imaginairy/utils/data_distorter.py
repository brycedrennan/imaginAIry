import contextlib
import math
import sys
from copy import deepcopy
from decimal import Decimal
from typing import Dict, Tuple, Union

NODE_DELETE = object()

DISTORTED_NUMBERS = [
    math.nan,
    math.inf,
    -math.inf,
    0,
    1,
    -1,
    0.000000000001,
    -0.000000000001,
    2**1024,
    -(2**1024),
    Decimal("0.000000000001"),
    Decimal(1) / Decimal(3),
    1.0 / 3.0,
    sys.float_info.max,
    "20",
    1.3333333333333333333e20,
]

DISTORTED_DATES = [
    "2021-01-01T00:00:00",
    "2021-01-01",
    "0000-00-00",
    "0001-01-01",
    "9001-01-01",
]

DISTORTED_STRINGS = [
    "",
    b"\00\001\002\003\004\005\006\007\010\011\012\013\014\015\016\017",
    " ",
    "\t\r\n",
    "\\r\\n\\t",
    "hello",
    "👩‍👩‍👧‍👧👩‍👩‍👧‍👧👩‍👩‍👧‍👧👩‍👩‍👧‍👧👩‍👩‍👧‍👧",
    "a" * 10000,
    "0",
    "!@#$%^&*()_+-=[]{}|;':\",.<>?/©™®",
    "你好こんにちは안녕하세요Привет",
    "<script>alert('Hello')</script>",
    (
        "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
        "ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħ"
        "ĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋ"
        "ŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲų"
        "ŴŵŶŷŸŹźŻżŽžſ"
    ),
]

DISTORTED_BOOLEAN = [
    True,
    False,
    "True",
    "False",
]

DISTORTED_OTHER = [(), object(), type(object), lambda x: x, NODE_DELETE, None]

DISTORTED_VALUES = (
    DISTORTED_NUMBERS
    + DISTORTED_DATES
    + DISTORTED_STRINGS
    + DISTORTED_BOOLEAN
    + DISTORTED_OTHER
)


class DataDistorter:
    def __init__(self, data, add_data_values=True):
        self.data = deepcopy(data)
        self.data_map, self.data_unique_values = create_node_map(self.data)
        self.distortion_values = [*DISTORTED_VALUES]
        if add_data_values:
            self.distortion_values += list(self.data_unique_values)

    def make_distorted_copy(self, node_number: int, distorted_value):
        """
        Make a distorted copy of the data.

        The node number is the index in the node map.
        """
        data = deepcopy(self.data)
        data = replace_value_at_path(data, self.data_map[node_number], distorted_value)
        return data

    def single_distortions(self):
        for node_number in range(len(self.data_map)):
            for distorted_value in DISTORTED_VALUES:
                yield self.make_distorted_copy(node_number, distorted_value)

    def double_distortions(self):
        for node_number in range(len(self.data_map)):
            for distorted_value in DISTORTED_VALUES:
                self.make_distorted_copy(node_number, distorted_value)

    def __iter__(self):
        for node_number in range(len(self.data_map)):
            for distorted_value in DISTORTED_VALUES:
                yield self.make_distorted_copy(node_number, distorted_value)


# nested dictionary helper functions


def create_node_map(data: Union[dict, list, tuple]) -> Tuple[Dict[int, list], set]:
    """
    Create a map of node numbers to paths in a nested dictionary.

    Include all nodes, not just leaves.

    Example:
    data = {"a": {"b": ["c", "d"]}, "e": "f"}
    node_map = create_node_map(data)
    assert node_map = {
        0: [],
        1: ["a"],
        2: ["a", "b"],
        3: ["a", "b", 0],
        4: ["a", "b", 1],
        5: ["e"],
    }

    """
    node_map = {}
    node_values = set()
    node_num = [
        0
    ]  # Using a list to hold the current node number as integers are immutable

    def _traverse(curr_data, curr_path):
        node_map[node_num[0]] = curr_path.copy()
        node_num[0] += 1

        if isinstance(curr_data, dict):
            for key, value in curr_data.items():
                _traverse(value, [*curr_path, key])
        elif isinstance(curr_data, (list, tuple)):
            for idx, item in enumerate(curr_data):
                _traverse(item, [*curr_path, idx])
        else:
            with contextlib.suppress(TypeError):
                node_values.add(curr_data)

    _traverse(data, [])
    return node_map, node_values


def get_path(data: dict, path):
    """Get a value from a nested dictionary using a path."""
    curr_data = data
    for key in path:
        curr_data = curr_data[key]
    return curr_data


def replace_value_at_path(data, path, new_value):
    """Replace a value in a nested dictionary using a path."""
    if not path:
        return new_value

    parent = get_path(data, path[:-1])
    last_key = path[-1]
    if new_value == NODE_DELETE:
        del parent[last_key]
    else:
        parent[last_key] = new_value
    return data

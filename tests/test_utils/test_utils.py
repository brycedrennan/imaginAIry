import platform
from datetime import datetime
from functools import lru_cache
from unittest import mock

import pytest
import torch.backends.mps
import torch.cuda

from imaginairy.utils import (
    get_device,
    get_hardware_description,
    get_obj_from_str,
    glob_expand_paths,
    instantiate_from_config,
)


def test_get_device(monkeypatch):
    # just run it for real to check that it doesn't error before we mock things
    get_device()

    m_cuda_is_available = mock.MagicMock()
    m_mps_is_available = mock.MagicMock()

    monkeypatch.setattr(torch.cuda, "is_available", m_cuda_is_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", m_mps_is_available)

    get_device.cache_clear()
    m_cuda_is_available.side_effect = lambda: True
    m_mps_is_available.side_effect = lambda: False
    assert get_device() == "cuda"

    get_device.cache_clear()
    m_cuda_is_available.side_effect = lambda: False
    m_mps_is_available.side_effect = lambda: True
    assert get_device() == "mps"

    get_device.cache_clear()
    m_cuda_is_available.side_effect = lambda: False
    m_mps_is_available.side_effect = lambda: False
    assert get_device() == "cpu"


def test_get_hardware_description(monkeypatch):
    get_hardware_description.cache_clear()
    monkeypatch.setattr(platform, "platform", lambda: "macOS-12.5.1-arm64-arm-64bit-z")
    assert get_hardware_description("cpu") == "macOS-12.5.1-arm64-arm-64bit-z"

    monkeypatch.setattr(platform, "platform", lambda: "macOS-12.5.1-arm64-arm-64bit-z")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda x: "rtx-3090")
    assert get_hardware_description("cuda") == "macOS-12.5.1-arm64-arm-64bit-z-rtx-3090"
    get_hardware_description.cache_clear()


def test_get_obj_from_str():
    foo = get_obj_from_str("functools.lru_cache")
    assert lru_cache == foo

    foo = get_obj_from_str("functools.lru_cache", reload=True)
    assert lru_cache != foo


def test_instantiate_from_config():
    config = {
        "target": "datetime.datetime",
        "params": {"year": 2002, "month": 10, "day": 1},
    }
    o = instantiate_from_config(config)
    assert o == datetime(2002, 10, 1)  # noqa: DTZ001

    config = "__is_first_stage__"
    assert instantiate_from_config(config) is None

    config = "__is_unconditional__"
    assert instantiate_from_config(config) is None

    config = "asdf"
    with pytest.raises(KeyError):
        instantiate_from_config(config)


class TestGlobExpandPaths:
    def test_valid_file_paths(self, tmp_path):
        # create temporary file
        file_path = tmp_path / "test.txt"
        file_path.touch()

        # test function with valid file path
        result = glob_expand_paths([str(file_path)])
        assert result == [str(file_path)]

    def test_valid_http_urls(self):
        # test function with valid http url
        result = glob_expand_paths(["http://www.example.com"])
        assert result == ["http://www.example.com"]

    def test_file_paths_with_wildcards(self, tmp_path):
        # create temporary files
        file1 = tmp_path / "test1.txt"
        file1.touch()
        file2 = tmp_path / "test2.txt"
        file2.touch()

        # test function with file path containing wildcard
        result = glob_expand_paths([str(tmp_path / "*.txt")])
        result.sort()
        assert result == [str(file1), str(file2)]

    def test_empty_input(self):
        # test function with empty input list
        result = glob_expand_paths([])
        assert not result

    def test_nonexistent_file_paths(self):
        # test function with non-existent file path
        result = glob_expand_paths(["/nonexistent/path"])
        assert not result

    def test_user_expansion(self, monkeypatch, tmp_path):
        file1 = tmp_path / "test1.txt"
        file1.touch()

        # monkeypatch os.path.expanduser to return a known path
        monkeypatch.setattr("os.path.expanduser", lambda x: str(tmp_path / "test1.txt"))

        # test function with user expansion
        paths = ["~/file.txt"]
        assert glob_expand_paths(paths) == [str(tmp_path / "test1.txt")]

import pytest
from packaging.version import Version

from imaginairy.utils.torch_installer import determine_torch_index

index_base = "https://download.pytorch.org/whl/"
index_cu118 = f"{index_base}cu118"
index_cu121 = f"{index_base}cu121"

torch_index_fixture = [
    (Version("11.8"), "linux", index_cu118),
    (Version("12.1"), "linux", ""),
    (Version("12.2"), "linux", ""),
    (Version("12.1"), "windows", index_cu121),
    (Version("12.2"), "windows", index_cu121),
]


@pytest.mark.parametrize(("cuda_version", "platform", "expected"), torch_index_fixture)
def test_determine_torch_index(cuda_version, platform, expected):
    assert determine_torch_index(cuda_version, platform) == expected

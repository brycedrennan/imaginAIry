import pytest
from packaging.version import Version

from imaginairy.utils.torch_installer import check_cuda_version


def test_check_cuda_version_ok():
    check_cuda_version("12.4")
    check_cuda_version("12.6")
    check_cuda_version("13.0")
    check_cuda_version(Version("12.4"))


def test_check_cuda_version_too_old():
    with pytest.raises(ValueError, match="too old"):
        check_cuda_version("12.3")
    with pytest.raises(ValueError, match="too old"):
        check_cuda_version("11.8")

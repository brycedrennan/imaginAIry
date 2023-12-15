"""Functions for managing PyTorch installation"""

import logging
import subprocess

from packaging.version import Version

logger = logging.getLogger(__name__)


def torch_version_check():
    if not could_install_better_torch_version():
        return

    import platform

    import torch

    from imaginairy.utils.debug_info import get_nvidia_smi_data

    nvidia_data = get_nvidia_smi_data()

    cuda_version = Version(nvidia_data["cuda_version"])
    cmd_parts = generate_torch_install_command(
        installed_cuda_version=cuda_version, system_type=platform.system().lower()
    )
    cmd_str = " ".join(cmd_parts)
    linebreak = "*" * 72
    msg = (
        f"\n{linebreak}\n"
        f"torch=={torch.__version__} is installed and unable to use CUDA {cuda_version}.\n\n"
        "You can install the correct version by running:\n\n"
        f"   pip uninstall torch torchvision -y\n"
        f"   {cmd_str}\n\n"
        "Installing the correct version will speed up image generation.\n"
        f"{linebreak}\n"
    )
    logger.warning(msg)


def could_install_better_torch_version():
    import platform

    if platform.system().lower() not in ("windows", "linux"):
        return False

    import torch

    if torch.cuda.is_available():
        return False

    from imaginairy.utils.debug_info import get_nvidia_smi_data

    nvidia_data = get_nvidia_smi_data()
    cuda_version = nvidia_data["cuda_version"]
    if cuda_version is None:
        return False

    cuda_version = Version(cuda_version)
    determine_torch_index(
        installed_cuda_version=cuda_version, system_type=platform.system()
    )

    return True


def determine_torch_index(installed_cuda_version: Version, system_type: str):
    cuda_pypi_base_url = "https://download.pytorch.org/whl/"
    min_required_cuda_version = Version("11.8")
    system_type = system_type.lower()

    if installed_cuda_version < min_required_cuda_version:
        msg = f"Your CUDA version ({installed_cuda_version}) is too old. Please upgrade to at least CUDA {min_required_cuda_version}."
        raise ValueError(msg)

    if system_type == "windows":
        if installed_cuda_version >= Version("12.1"):
            return f"{cuda_pypi_base_url}cu121"
        if installed_cuda_version >= Version("12.0"):
            raise ValueError("You should upgrade to CUDA>=12.1")
        if installed_cuda_version >= Version("11.8"):
            return f"{cuda_pypi_base_url}cu118"

    elif system_type == "linux":
        if installed_cuda_version >= Version("12.1"):
            return ""
        if installed_cuda_version >= Version("12.0"):
            raise ValueError("You should upgrade to CUDA>=12.1")
        if installed_cuda_version >= Version("11.8"):
            return f"{cuda_pypi_base_url}cu118"

    return ""


def generate_torch_install_command(installed_cuda_version: Version, system_type):
    packages = ["torch", "torchvision"]
    index_url = determine_torch_index(
        installed_cuda_version=installed_cuda_version, system_type=system_type
    )
    cmd_parts = [
        "pip",
        "install",
        "--upgrade",
    ]
    if index_url:
        cmd_parts.extend(
            [
                "--index-url",
                index_url,
            ]
        )
    cmd_parts.extend(packages)
    return cmd_parts


def install_packages(packages, index_url):
    """
    Install a list of Python packages from a specified index server.

    :param packages: A list of package names to install.
    :param index_url: The URL of the Python package index server.
    """
    import sys

    for package in packages:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package,
                    "--index-url",
                    index_url,
                ]
            )
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

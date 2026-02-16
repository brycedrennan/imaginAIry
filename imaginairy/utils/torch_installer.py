"""Functions for managing PyTorch installation"""

import logging

from packaging.version import Version

logger = logging.getLogger(__name__)

MIN_CUDA_VERSION = Version("12.4")


def torch_version_check():
    if not could_install_better_torch_version():
        return

    import torch

    from imaginairy.utils.debug_info import get_nvidia_smi_data

    nvidia_data = get_nvidia_smi_data()
    cuda_version = nvidia_data["cuda_version"]
    linebreak = "*" * 72
    msg = (
        f"\n{linebreak}\n"
        f"torch=={torch.__version__} is installed and unable to use CUDA {cuda_version}.\n\n"
        "You can install the correct version by running:\n\n"
        "   pip uninstall torch torchvision -y\n"
        "   pip install --upgrade torch torchvision\n\n"
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
    return not cuda_version < MIN_CUDA_VERSION


def check_cuda_version(cuda_version: str | Version):
    """Raise ValueError if CUDA version is below minimum."""
    if not isinstance(cuda_version, Version):
        cuda_version = Version(cuda_version)
    if cuda_version < MIN_CUDA_VERSION:
        msg = f"Your CUDA version ({cuda_version}) is too old. Please upgrade to at least CUDA {MIN_CUDA_VERSION}."
        raise ValueError(msg)

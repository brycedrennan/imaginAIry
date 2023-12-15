"""Functions for gathering system debug information"""

from functools import lru_cache


def get_debug_info():
    import os.path
    import platform
    import sys

    import psutil
    import torch

    from imaginairy.utils import get_device, get_hardware_description
    from imaginairy.version import get_version

    data = {
        "imaginairy_version": get_version(),
        "imaginairy_path": os.path.dirname(__file__),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_installation_path": sys.executable,
        "device": get_device(),
        "torch_version": torch.__version__,
        "platform": platform.system(),
        "hardware_description": get_hardware_description(get_device()),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        data.update(
            {
                "cuda_version": torch.version.cuda,
                "graphics_card": device_props.name,
                "graphics_card_memory": device_props.total_memory,
                "graphics_card_processor_count": device_props.multi_processor_count,
                "graphics_card_hw_version": f"{device_props.major}.{device_props.minor}",
            }
        )
    elif "mps" in get_device():
        data.update(
            {
                "graphics_card": "Apple MPS",
            }
        )

    nvidia_data = get_nvidia_smi_data()
    data.update(nvidia_data)

    return data


@lru_cache
def _get_nvidia_smi_output():
    import subprocess

    try:
        output = subprocess.check_output(
            "nvidia-smi", text=True, shell=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        output = "no nvidia card found"

    return output


def _process_nvidia_smi_output(output):
    import re

    cuda_version = re.search(r"CUDA Version: (\d+\.\d+)", output)

    return {
        "cuda_version": cuda_version.group(1) if cuda_version else None,
        "nvidia_smi": output,
    }


def get_nvidia_smi_data():
    smi_output = _get_nvidia_smi_output()
    return _process_nvidia_smi_output(smi_output)

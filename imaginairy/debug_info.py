import os.path
import sys

import torch

from imaginairy import __version__
from imaginairy.utils import get_device, get_hardware_description


def get_debug_info():
    data = {
        "imaginairy_version": __version__,
        "imaginairy_path": os.path.dirname(__file__),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_installation_path": sys.executable,
        "device": get_device(),
        "torch_version": torch.__version__,
        "platform": sys.platform,
        "hardware_description": get_hardware_description(get_device()),
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
    return data

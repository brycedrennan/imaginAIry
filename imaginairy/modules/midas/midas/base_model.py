"""Classes for loading model weights"""

import torch

from imaginairy.utils.model_manager import get_cached_url_path


class BaseModel(torch.nn.Module):
    def load(self, path):
        """
        Load model from file.

        Args:
            path (str): file path
        """
        ckpt_path = get_cached_url_path(path, category="weights")
        parameters = torch.load(ckpt_path, map_location=torch.device("cpu"))
        parameters = {
            k: v for k, v in parameters.items() if "relative_position_index" not in k
        }
        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

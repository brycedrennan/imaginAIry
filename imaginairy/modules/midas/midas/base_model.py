import torch

from imaginairy import config
from imaginairy.model_manager import get_cached_url_path


class BaseModel(torch.nn.Module):
    def load(self, path):
        """
        Load model from file.

        Args:
            path (str): file path
        """
        ckpt_path = get_cached_url_path(config.midas_url, category="weights")
        parameters = torch.load(ckpt_path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

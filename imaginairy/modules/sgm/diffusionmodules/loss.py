from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from imaginairy.modules.sgm.autoencoding.lpips.loss.lpips import LPIPS
from imaginairy.modules.sgm.encoders.modules import GeneralConditioner
from imaginairy.utils import instantiate_from_config
from imaginairy.vendored.k_diffusion.utils import append_dims

from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input_tensor + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input_tensor: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input_tensor, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input_tensor: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input_tensor.shape[0]).to(input)

        noise = torch.randn_like(input_tensor)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input_tensor.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input_tensor.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input_tensor.device),
                input_tensor.ndim,
            )
        sigmas_bc = append_dims(sigmas, input_tensor.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input_tensor)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input_tensor.ndim)
        return self.get_loss(model_output, input_tensor, w)

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            msg = f"Unknown loss type {self.loss_type}"
            raise NotImplementedError(msg)

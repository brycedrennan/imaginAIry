import pytest
import torch

from imaginairy.utils import get_device


@pytest.mark.skipif("mps" not in get_device(), reason="MPS only bug")
@pytest.mark.xfail(reason="MPS only bug")
def test_sigma_bug():
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/4558#issuecomment-1310387114
    def t_fn_a(sigma):
        return sigma.to(get_device()).log().neg()

    def t_fn_b(sigma):
        return sigma.to("cpu").log().neg().to(get_device())

    sigmas = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device=get_device())

    for i in range(sigmas.size()[0]):
        assert t_fn_a(sigmas[i]) == t_fn_b(sigmas[i])

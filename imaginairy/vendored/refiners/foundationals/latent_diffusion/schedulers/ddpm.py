from torch import Tensor, arange, device as Device

from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler


class DDPM(Scheduler):
    """
    The Denoising Diffusion Probabilistic Models (DDPM) is a specific type of diffusion model,
    which uses a specific strategy to generate the timesteps and applies the diffusion process in a specific way.
    """

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        device: Device | str = "cpu",
    ) -> None:
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            device=device,
        )

    def _generate_timesteps(self) -> Tensor:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = arange(start=0, end=self.num_inference_steps, step=1, device=self.device) * step_ratio
        return timesteps.flip(0)

    def __call__(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        raise NotImplementedError

from torch import Generator, Tensor, arange, device as Device

from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler


class DDPM(Scheduler):
    """
    Denoising Diffusion Probabilistic Model

    Only used for training Latent Diffusion models. Cannot be called.
    """

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
    ) -> None:
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            first_inference_step=first_inference_step,
            device=device,
        )

    def _generate_timesteps(self) -> Tensor:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = arange(start=0, end=self.num_inference_steps, step=1, device=self.device) * step_ratio
        return timesteps.flip(0)

    def __call__(self, x: Tensor, noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        raise NotImplementedError

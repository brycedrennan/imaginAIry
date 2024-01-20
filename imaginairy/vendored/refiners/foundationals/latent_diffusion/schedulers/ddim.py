from torch import Generator, Tensor, arange, device as Device, dtype as Dtype, float32, sqrt, tensor

from imaginairy.vendored.refiners.foundationals.latent_diffusion.schedulers.scheduler import NoiseSchedule, Scheduler


class DDIM(Scheduler):
    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
        dtype: Dtype = float32,
    ) -> None:
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            noise_schedule=noise_schedule,
            first_inference_step=first_inference_step,
            device=device,
            dtype=dtype,
        )
        self.timesteps = self._generate_timesteps()

    def _generate_timesteps(self) -> Tensor:
        """
        Generates decreasing timesteps with 'leading' spacing and offset of 1
        similar to diffusers settings for the DDIM scheduler in Stable Diffusion 1.5
        """
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = arange(start=0, end=self.num_inference_steps, step=1, device=self.device) * step_ratio + 1
        return timesteps.flip(0)

    def __call__(self, x: Tensor, noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        assert self.first_inference_step <= step < self.num_inference_steps, "invalid step {step}"

        timestep, previous_timestep = (
            self.timesteps[step],
            (
                self.timesteps[step + 1]
                if step < self.num_inference_steps - 1
                else tensor(data=[0], device=self.device, dtype=self.dtype)
            ),
        )
        current_scale_factor, previous_scale_factor = (
            self.cumulative_scale_factors[timestep],
            (
                self.cumulative_scale_factors[previous_timestep]
                if previous_timestep > 0
                else self.cumulative_scale_factors[0]
            ),
        )
        predicted_x = (x - sqrt(1 - current_scale_factor**2) * noise) / current_scale_factor
        noise_factor = sqrt(1 - previous_scale_factor**2)

        # Do not add noise at the last step to avoid visual artifacts.
        if step == self.num_inference_steps - 1:
            noise_factor = 0

        denoised_x = previous_scale_factor * predicted_x + noise_factor * noise

        return denoised_x

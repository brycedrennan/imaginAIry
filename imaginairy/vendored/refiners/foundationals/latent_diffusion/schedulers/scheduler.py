from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar

from torch import Generator, Tensor, device as Device, dtype as DType, float32, linspace, log, sqrt

T = TypeVar("T", bound="Scheduler")


class NoiseSchedule(str, Enum):
    UNIFORM = "uniform"
    QUADRATIC = "quadratic"
    KARRAS = "karras"


class Scheduler(ABC):
    """
    A base class for creating a diffusion model scheduler.

    The Scheduler creates a sequence of noise and scaling factors used in the diffusion process,
    which gradually transforms the original data distribution into a Gaussian one.

    This process is described using several parameters such as initial and final diffusion rates,
    and is encapsulated into a `__call__` method that applies a step of the diffusion process.
    """

    timesteps: Tensor

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
        dtype: DType = float32,
    ):
        self.device: Device = Device(device)
        self.dtype: DType = dtype
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.initial_diffusion_rate = initial_diffusion_rate
        self.final_diffusion_rate = final_diffusion_rate
        self.noise_schedule = noise_schedule
        self.first_inference_step = first_inference_step
        self.scale_factors = self.sample_noise_schedule()
        self.cumulative_scale_factors = sqrt(self.scale_factors.cumprod(dim=0))
        self.noise_std = sqrt(1.0 - self.scale_factors.cumprod(dim=0))
        self.signal_to_noise_ratios = log(self.cumulative_scale_factors) - log(self.noise_std)
        self.timesteps = self._generate_timesteps()

    @abstractmethod
    def __call__(self, x: Tensor, noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """
        Applies a step of the diffusion process to the input tensor `x` using the provided `noise` and `timestep`.

        This method should be overridden by subclasses to implement the specific diffusion process.
        """
        ...

    @abstractmethod
    def _generate_timesteps(self) -> Tensor:
        """
        Generates a tensor of timesteps.

        This method should be overridden by subclasses to provide the specific timesteps for the diffusion process.
        """
        ...

    @property
    def all_steps(self) -> list[int]:
        return list(range(self.num_inference_steps))

    @property
    def inference_steps(self) -> list[int]:
        return self.all_steps[self.first_inference_step :]

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        """
        For compatibility with schedulers that need to scale the input according to the current timestep.
        """
        return x

    def sample_power_distribution(self, power: float = 2, /) -> Tensor:
        return (
            linspace(
                start=self.initial_diffusion_rate ** (1 / power),
                end=self.final_diffusion_rate ** (1 / power),
                steps=self.num_train_timesteps,
                device=self.device,
                dtype=self.dtype,
            )
            ** power
        )

    def sample_noise_schedule(self) -> Tensor:
        match self.noise_schedule:
            case "uniform":
                return 1 - self.sample_power_distribution(1)
            case "quadratic":
                return 1 - self.sample_power_distribution(2)
            case "karras":
                return 1 - self.sample_power_distribution(7)
            case _:
                raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def add_noise(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
    ) -> Tensor:
        timestep = self.timesteps[step]
        cumulative_scale_factors = self.cumulative_scale_factors[timestep]
        noise_stds = self.noise_std[timestep]
        noised_x = cumulative_scale_factors * x + noise_stds * noise
        return noised_x

    def remove_noise(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        timestep = self.timesteps[step]
        cumulative_scale_factors = self.cumulative_scale_factors[timestep]
        noise_stds = self.noise_std[timestep]
        # See equation (15) from https://arxiv.org/pdf/2006.11239.pdf. Useful to preview progress or for guidance like
        # in https://arxiv.org/pdf/2210.00939.pdf (self-attention guidance)
        denoised_x = (x - noise_stds * noise) / cumulative_scale_factors
        return denoised_x

    def to(self: T, device: Device | str | None = None, dtype: DType | None = None) -> T:  # type: ignore
        if device is not None:
            self.device = Device(device)
            self.timesteps = self.timesteps.to(device)
        if dtype is not None:
            self.dtype = dtype
        self.scale_factors = self.scale_factors.to(device, dtype=dtype)
        self.cumulative_scale_factors = self.cumulative_scale_factors.to(device, dtype=dtype)
        self.noise_std = self.noise_std.to(device, dtype=dtype)
        self.signal_to_noise_ratios = self.signal_to_noise_ratios.to(device, dtype=dtype)
        return self

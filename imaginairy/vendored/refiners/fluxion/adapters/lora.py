from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, device as Device, dtype as DType
from torch.nn import Parameter as TorchParameter
from torch.nn.init import normal_, zeros_

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.layers.chain import Chain


class Lora(fl.Chain, ABC):
    def __init__(
        self,
        rank: int = 16,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.rank = rank
        self._scale = scale

        super().__init__(*self.lora_layers(device=device, dtype=dtype), fl.Multiply(scale))

        normal_(tensor=self.down.weight, std=1 / self.rank)
        zeros_(tensor=self.up.weight)

    @abstractmethod
    def lora_layers(
        self, device: Device | str | None = None, dtype: DType | None = None
    ) -> tuple[fl.WeightedModule, fl.WeightedModule]:
        ...

    @property
    def down(self) -> fl.WeightedModule:
        down_layer = self[0]
        assert isinstance(down_layer, fl.WeightedModule)
        return down_layer

    @property
    def up(self) -> fl.WeightedModule:
        up_layer = self[1]
        assert isinstance(up_layer, fl.WeightedModule)
        return up_layer

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value

    @classmethod
    def from_weights(
        cls,
        down: Tensor,
        up: Tensor,
    ) -> "Lora":
        match (up.ndim, down.ndim):
            case (2, 2):
                return LinearLora.from_weights(up=up, down=down)
            case (4, 4):
                return Conv2dLora.from_weights(up=up, down=down)
            case _:
                raise ValueError(f"Unsupported weight shapes: up={up.shape}, down={down.shape}")

    @classmethod
    def from_dict(cls, state_dict: dict[str, Tensor], /) -> dict[str, "Lora"]:
        """
        Create a dictionary of LoRA layers from a state dict.

        Expects the state dict to be a succession of down and up weights.
        """
        state_dict = {k: v for k, v in state_dict.items() if ".weight" in k}
        loras: dict[str, Lora] = {}
        for down_key, down_tensor, up_tensor in zip(
            list(state_dict.keys())[::2], list(state_dict.values())[::2], list(state_dict.values())[1::2]
        ):
            key = ".".join(down_key.split(".")[:-2])
            loras[key] = cls.from_weights(down=down_tensor, up=up_tensor)
        return loras

    @abstractmethod
    def auto_attach(self, target: fl.Chain, exclude: list[str] | None = None) -> Any:
        ...

    def load_weights(self, down_weight: Tensor, up_weight: Tensor) -> None:
        assert down_weight.shape == self.down.weight.shape
        assert up_weight.shape == self.up.weight.shape
        self.down.weight = TorchParameter(down_weight.to(device=self.device, dtype=self.dtype))
        self.up.weight = TorchParameter(up_weight.to(device=self.device, dtype=self.dtype))


class LinearLora(Lora):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        super().__init__(rank=rank, scale=scale, device=device, dtype=dtype)

    @classmethod
    def from_weights(
        cls,
        down: Tensor,
        up: Tensor,
    ) -> "LinearLora":
        assert up.ndim == 2 and down.ndim == 2
        assert down.shape[0] == up.shape[1], f"Rank mismatch: down rank={down.shape[0]} and up rank={up.shape[1]}"
        lora = cls(
            in_features=down.shape[1], out_features=up.shape[0], rank=down.shape[0], device=up.device, dtype=up.dtype
        )
        lora.load_weights(down_weight=down, up_weight=up)
        return lora

    def auto_attach(self, target: Chain, exclude: list[str] | None = None) -> "tuple[LoraAdapter, fl.Chain] | None":
        for layer, parent in target.walk(fl.Linear):
            if isinstance(parent, Lora) or isinstance(parent, LoraAdapter):
                continue

            if exclude is not None and any(
                [any([p.__class__.__name__ == e for p in parent.get_parents() + [parent]]) for e in exclude]
            ):
                continue

            if layer.in_features == self.in_features and layer.out_features == self.out_features:
                return LoraAdapter(target=layer, lora=self), parent

    def lora_layers(
        self, device: Device | str | None = None, dtype: DType | None = None
    ) -> tuple[fl.Linear, fl.Linear]:
        return (
            fl.Linear(
                in_features=self.in_features,
                out_features=self.rank,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.Linear(
                in_features=self.rank,
                out_features=self.out_features,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )


class Conv2dLora(Lora):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 16,
        scale: float = 1.0,
        kernel_size: tuple[int, int] = (1, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 1),
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        super().__init__(rank=rank, scale=scale, device=device, dtype=dtype)

    @classmethod
    def from_weights(
        cls,
        down: Tensor,
        up: Tensor,
    ) -> "Conv2dLora":
        assert up.ndim == 4 and down.ndim == 4
        assert down.shape[0] == up.shape[1], f"Rank mismatch: down rank={down.shape[0]} and up rank={up.shape[1]}"
        down_kernel_size, up_kernel_size = down.shape[2], up.shape[2]
        down_padding = 1 if down_kernel_size == 3 else 0
        up_padding = 1 if up_kernel_size == 3 else 0
        lora = cls(
            in_channels=down.shape[1],
            out_channels=up.shape[0],
            rank=down.shape[0],
            kernel_size=(down_kernel_size, up_kernel_size),
            padding=(down_padding, up_padding),
            device=up.device,
            dtype=up.dtype,
        )
        lora.load_weights(down_weight=down, up_weight=up)
        return lora

    def auto_attach(self, target: Chain, exclude: list[str] | None = None) -> "tuple[LoraAdapter, fl.Chain]  | None":
        for layer, parent in target.walk(fl.Conv2d):
            if isinstance(parent, Lora) or isinstance(parent, LoraAdapter):
                continue

            if exclude is not None and any(
                [any([p.__class__.__name__ == e for p in parent.get_parents() + [parent]]) for e in exclude]
            ):
                continue

            if layer.in_channels == self.in_channels and layer.out_channels == self.out_channels:
                if layer.stride != (self.stride[0], self.stride[0]):
                    self.down.stride = layer.stride

                return LoraAdapter(
                    target=layer,
                    lora=self,
                ), parent

    def lora_layers(
        self, device: Device | str | None = None, dtype: DType | None = None
    ) -> tuple[fl.Conv2d, fl.Conv2d]:
        return (
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.rank,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
                use_bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.Conv2d(
                in_channels=self.rank,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
                use_bias=False,
                device=device,
                dtype=dtype,
            ),
        )


class LoraAdapter(fl.Sum, Adapter[fl.WeightedModule]):
    def __init__(self, target: fl.WeightedModule, lora: Lora) -> None:
        with self.setup_adapter(target):
            super().__init__(target, lora)

    @property
    def lora(self) -> Lora:
        return self.ensure_find(Lora)

    @property
    def scale(self) -> float:
        return self.lora.scale

    @scale.setter
    def scale(self, value: float) -> None:
        self.lora.scale = value

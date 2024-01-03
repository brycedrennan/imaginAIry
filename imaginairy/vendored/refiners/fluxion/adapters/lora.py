from typing import Any, Generic, Iterable, TypeVar

from torch import Tensor, device as Device, dtype as DType
from torch.nn import Parameter as TorchParameter
from torch.nn.init import normal_, zeros_

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter

T = TypeVar("T", bound=fl.Chain)
TLoraAdapter = TypeVar("TLoraAdapter", bound="LoraAdapter[Any]")  # Self (see PEP 673)


class Lora(fl.Chain):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale: float = 1.0

        super().__init__(
            fl.Linear(in_features=in_features, out_features=rank, bias=False, device=device, dtype=dtype),
            fl.Linear(in_features=rank, out_features=out_features, bias=False, device=device, dtype=dtype),
            fl.Lambda(func=self.scale_outputs),
        )

        normal_(tensor=self.Linear_1.weight, std=1 / self.rank)
        zeros_(tensor=self.Linear_2.weight)

    def scale_outputs(self, x: Tensor) -> Tensor:
        return x * self.scale

    def set_scale(self, scale: float) -> None:
        self.scale = scale

    def load_weights(self, down_weight: Tensor, up_weight: Tensor) -> None:
        self.Linear_1.weight = TorchParameter(down_weight.to(device=self.device, dtype=self.dtype))
        self.Linear_2.weight = TorchParameter(up_weight.to(device=self.device, dtype=self.dtype))

    @property
    def up_weight(self) -> Tensor:
        return self.Linear_2.weight.data

    @property
    def down_weight(self) -> Tensor:
        return self.Linear_1.weight.data


class SingleLoraAdapter(fl.Sum, Adapter[fl.Linear]):
    def __init__(
        self,
        target: fl.Linear,
        rank: int = 16,
        scale: float = 1.0,
    ) -> None:
        self.in_features = target.in_features
        self.out_features = target.out_features
        self.rank = rank
        self.scale = scale
        with self.setup_adapter(target):
            super().__init__(
                target,
                Lora(
                    in_features=target.in_features,
                    out_features=target.out_features,
                    rank=rank,
                    device=target.device,
                    dtype=target.dtype,
                ),
            )
        self.Lora.set_scale(scale=scale)


class LoraAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(
        self,
        target: T,
        sub_targets: Iterable[tuple[fl.Linear, fl.Chain]],
        rank: int | None = None,
        scale: float = 1.0,
        weights: list[Tensor] | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        if weights is not None:
            assert len(weights) % 2 == 0
            weights_rank = weights[0].shape[1]
            if rank is None:
                rank = weights_rank
            else:
                assert rank == weights_rank

        assert rank is not None, "either pass a rank or weights"

        self.sub_targets = sub_targets
        self.sub_adapters: list[tuple[SingleLoraAdapter, fl.Chain]] = []

        for linear, parent in self.sub_targets:
            self.sub_adapters.append((SingleLoraAdapter(target=linear, rank=rank, scale=scale), parent))

        if weights is not None:
            assert len(self.sub_adapters) == (len(weights) // 2)
            for i, (adapter, _) in enumerate(self.sub_adapters):
                lora = adapter.Lora
                assert (
                    lora.rank == weights[i * 2].shape[1]
                ), f"Rank of Lora layer {lora.rank} must match shape of weights {weights[i*2].shape[1]}"
                adapter.Lora.load_weights(up_weight=weights[i * 2], down_weight=weights[i * 2 + 1])

    def inject(self: TLoraAdapter, parent: fl.Chain | None = None) -> TLoraAdapter:
        for adapter, adapter_parent in self.sub_adapters:
            adapter.inject(adapter_parent)
        return super().inject(parent)

    def eject(self) -> None:
        for adapter, _ in self.sub_adapters:
            adapter.eject()
        super().eject()

    @property
    def weights(self) -> list[Tensor]:
        return [w for adapter, _ in self.sub_adapters for w in [adapter.Lora.up_weight, adapter.Lora.down_weight]]

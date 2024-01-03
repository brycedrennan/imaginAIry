from typing import Callable

from torch import Tensor

from imaginairy.vendored.refiners.fluxion.adapters.adapter import Adapter
from imaginairy.vendored.refiners.fluxion.layers import (
    Chain,
    Concatenate,
    Identity,
    Lambda,
    Parallel,
    Passthrough,
    SelfAttention,
    SetContext,
    UseContext,
)
from imaginairy.vendored.refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock
from imaginairy.vendored.refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet


class SaveLayerNormAdapter(Chain, Adapter[SelfAttention]):
    def __init__(self, target: SelfAttention, context: str) -> None:
        self.context = context
        with self.setup_adapter(target):
            super().__init__(SetContext(self.context, "norm"), target)


class SelfAttentionInjectionAdapter(Chain, Adapter[SelfAttention]):
    def __init__(
        self,
        target: SelfAttention,
        context: str,
        style_cfg: float = 0.5,
    ) -> None:
        self.context = context
        self.style_cfg = style_cfg

        sa_guided = target.structural_copy()
        assert isinstance(sa_guided[0], Parallel)
        sa_guided.replace(
            sa_guided[0],
            Parallel(
                Identity(),
                Concatenate(Identity(), UseContext(self.context, "norm"), dim=1),
                Concatenate(Identity(), UseContext(self.context, "norm"), dim=1),
            ),
        )

        with self.setup_adapter(target):
            slice_tensor: Callable[[Tensor], Tensor] = lambda x: x[:1]
            super().__init__(
                Parallel(sa_guided, Chain(Lambda(slice_tensor), target)),
                Lambda(self.compute_averaged_unconditioned_x),
            )

    def compute_averaged_unconditioned_x(self, x: Tensor, unguided_unconditioned_x: Tensor) -> Tensor:
        x[0] = self.style_cfg * x[0] + (1.0 - self.style_cfg) * unguided_unconditioned_x
        return x


class SelfAttentionInjectionPassthrough(Passthrough):
    def __init__(self, target: SD1UNet) -> None:
        guide_unet = target.structural_copy()
        for i, attention_block in enumerate(guide_unet.layers(CrossAttentionBlock)):
            sa = attention_block.ensure_find(SelfAttention)
            assert sa.parent is not None
            SaveLayerNormAdapter(sa, context=f"self_attention_context_{i}").inject()

        super().__init__(
            Lambda(self._copy_diffusion_context),
            UseContext("reference_only_control", "guide"),
            guide_unet,
            Lambda(self._restore_diffusion_context),
        )

    def _copy_diffusion_context(self, x: Tensor) -> Tensor:
        # This function allows to not disrupt the accumulation of residuals in the unet (if controlnet are used)
        self.set_context(
            "self_attention_residuals_buffer",
            {"buffer": self.use_context("unet")["residuals"]},
        )
        self.set_context(
            "unet",
            {"residuals": [0.0] * 13},
        )
        return x

    def _restore_diffusion_context(self, x: Tensor) -> Tensor:
        self.set_context(
            "unet",
            {
                "residuals": self.use_context("self_attention_residuals_buffer")["buffer"],
            },
        )
        return x


class ReferenceOnlyControlAdapter(Chain, Adapter[SD1UNet]):
    # TODO: Does not support batching yet. Assumes concatenated inputs for classifier-free guidance

    def __init__(self, target: SD1UNet, style_cfg: float = 0.5) -> None:
        # the style_cfg is the weight of the guide in unconditionned diffusion.
        # This value is recommended to be 0.5 on the sdwebui repo.

        self.sub_adapters: list[SelfAttentionInjectionAdapter] = []
        self._passthrough: list[SelfAttentionInjectionPassthrough] = [
            SelfAttentionInjectionPassthrough(target)
        ]  # not registered by PyTorch

        with self.setup_adapter(target):
            super().__init__(target)

        for i, attention_block in enumerate(target.layers(CrossAttentionBlock)):
            self.set_context(f"self_attention_context_{i}", {"norm": None})

            sa = attention_block.ensure_find(SelfAttention)
            assert sa.parent is not None

            self.sub_adapters.append(
                SelfAttentionInjectionAdapter(sa, context=f"self_attention_context_{i}", style_cfg=style_cfg)
            )

    def inject(self: "ReferenceOnlyControlAdapter", parent: Chain | None = None) -> "ReferenceOnlyControlAdapter":
        passthrough = self._passthrough[0]
        assert passthrough not in self.target, f"{passthrough} is already injected"
        for adapter in self.sub_adapters:
            adapter.inject()
        self.target.insert(0, passthrough)
        return super().inject(parent)

    def eject(self) -> None:
        passthrough = self._passthrough[0]
        assert self.target[0] == passthrough, f"{passthrough} is not the first element of target UNet"
        for adapter in self.sub_adapters:
            adapter.eject()
        self.target.pop(0)
        super().eject()

    def set_controlnet_condition(self, condition: Tensor) -> None:
        self.set_context("reference_only_control", {"guide": condition})

    def structural_copy(self: "ReferenceOnlyControlAdapter") -> "ReferenceOnlyControlAdapter":
        raise RuntimeError("ReferenceOnlyControlAdapter cannot be copied, eject it first.")

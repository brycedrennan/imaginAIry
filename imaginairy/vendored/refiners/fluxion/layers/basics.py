import torch
from torch import Size, Tensor, device as Device, dtype as DType, randn
from torch.nn import Parameter as TorchParameter

from imaginairy.vendored.refiners.fluxion.layers.module import Module, WeightedModule


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class View(Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*self.shape)


class GetArg(Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, *args: Tensor) -> Tensor:
        return args[self.index]


class Flatten(Module):
    def __init__(self, start_dim: int = 0, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, sizes: Size) -> Tensor:
        return x.unflatten(self.dim, sizes)  # type: ignore


class Reshape(Module):
    """
    Reshape the input tensor to the given shape. The shape must be compatible with the input tensor shape. The batch
    dimension is preserved.
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.shape)


class Transpose(Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)


class Permute(Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)


class Slicing(Module):
    def __init__(self, dim: int = 0, start: int = 0, end: int | None = None, step: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x: Tensor) -> Tensor:
        dim_size = x.shape[self.dim]
        start = self.start if self.start >= 0 else dim_size + self.start
        end = self.end or dim_size
        end = end if end >= 0 else dim_size + end
        start = max(min(start, dim_size), 0)
        end = max(min(end, dim_size), 0)
        if start >= end:
            return self.get_empty_slice(x)
        indices = torch.arange(start=start, end=end, step=self.step, device=x.device)
        return x.index_select(self.dim, indices)

    def get_empty_slice(self, x: Tensor) -> Tensor:
        """
        Return an empty slice of the same shape as the input tensor to mimic PyTorch's slicing behavior.
        """
        shape = list(x.shape)
        shape[self.dim] = 0
        return torch.empty(*shape, device=x.device)


class Squeeze(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(self.dim)


class Unsqueeze(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.dim)


class Unbind(Module):
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
        super().__init__()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return x.unbind(dim=self.dim)  # type: ignore


class Chunk(Module):
    def __init__(self, chunks: int, dim: int = 0) -> None:
        self.chunks = chunks
        self.dim = dim
        super().__init__()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return x.chunk(chunks=self.chunks, dim=self.dim)  # type: ignore


class Sin(Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(input=x)


class Cos(Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(input=x)


class Multiply(Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class Parameter(WeightedModule):
    """
    A layer that wraps a tensor as a parameter. This is useful to create a parameter that is not a weight or a bias.
    """

    def __init__(self, *dims: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__()
        self.dims = dims
        self.weight = TorchParameter(randn(*dims, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight.expand(x.shape[0], *self.dims)


class Buffer(WeightedModule):
    """
    A layer that wraps a tensor as a buffer. This is useful to create a buffer that is not a weight or a bias.

    Buffers are not trainable.
    """

    def __init__(self, *dims: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__()
        self.dims = dims
        self.register_buffer("buffer", randn(*dims, device=device, dtype=dtype))

    @property
    def device(self) -> Device:
        return self.buffer.device

    @property
    def dtype(self) -> DType:
        return self.buffer.dtype

    def forward(self, _: Tensor) -> Tensor:
        return self.buffer

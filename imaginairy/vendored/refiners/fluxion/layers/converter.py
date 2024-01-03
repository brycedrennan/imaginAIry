from torch import Tensor

from imaginairy.vendored.refiners.fluxion.layers.module import ContextModule


class Converter(ContextModule):
    """
    A Converter class that adjusts tensor properties based on a parent module's settings.

    This class inherits from `ContextModule` and provides functionality to adjust
    the device and dtype of input tensor(s) to match the parent module's attributes.

    Attributes:
        set_device (bool): If True, matches the device of the input tensor(s) to the parent's device.
        set_dtype (bool): If True, matches the dtype of the input tensor(s) to the parent's dtype.

    Note:
        Ensure the parent module has `device` and `dtype` attributes if `set_device` or `set_dtype` are set to True.
    """

    def __init__(self, set_device: bool = True, set_dtype: bool = True) -> None:
        super().__init__()
        self.set_device = set_device
        self.set_dtype = set_dtype

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        parent = self.ensure_parent
        converted_tensors: list[Tensor] = []

        for x in inputs:
            if self.set_device:
                device = parent.device
                assert device is not None, "parent has no device"
                x = x.to(device=device)
            if self.set_dtype:
                dtype = parent.dtype
                assert dtype is not None, "parent has no dtype"
                x = x.to(dtype=dtype)

            converted_tensors.append(x)

        return tuple(converted_tensors)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(set_device={self.set_device}, set_dtype={self.set_dtype})"

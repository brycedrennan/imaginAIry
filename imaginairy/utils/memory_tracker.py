import contextlib
from typing import Callable, List

import torch


class TorchRAMTracker(contextlib.ContextDecorator):
    """Tracks peak CUDA memory usage for a block of code."""

    _memory_stack: List[int] = []
    mem_interface = torch.cuda

    def __init__(
        self, name="", callback_fn: "Callable[[TorchRAMTracker], None] | None" = None
    ):
        self.name = name
        self.peak_memory = 0
        self.start_memory = 0
        self.end_memory = 0
        self.callback_fn = callback_fn
        self._stack_depth = None

    def start(self):
        current_peak = self.mem_interface.max_memory_allocated()
        TorchRAMTracker._memory_stack.append(current_peak)
        self._stack_depth = len(TorchRAMTracker._memory_stack)
        self.mem_interface.reset_peak_memory_stats()
        self.start_memory = self.mem_interface.memory_allocated()

    def stop(self):
        end_peak = self.mem_interface.max_memory_allocated()
        peaks = TorchRAMTracker._memory_stack[self._stack_depth :] + [end_peak]
        self.peak_memory = max(peaks)
        del TorchRAMTracker._memory_stack[self._stack_depth :]
        self.end_memory = self.mem_interface.memory_allocated()
        self.peak_memory_delta = self.peak_memory - self.start_memory

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

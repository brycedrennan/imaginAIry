# pylama: ignore=W0212
import logging
from collections import OrderedDict
from functools import cached_property

from imaginairy.utils import get_device

logger = logging.getLogger(__name__)


log = logger.debug


def get_model_size(model):
    from torch import nn

    if not isinstance(model, nn.Module) and hasattr(model, "model"):
        model = model.model

    return sum(v.numel() * v.element_size() for v in model.parameters())


def move_model_device(model, device):
    from torch import nn

    if not isinstance(model, nn.Module) and hasattr(model, "model"):
        model = model.model

    return model.to(device)


class MemoryTrackingCache:
    def __init__(self, *args, **kwargs):
        self.memory_usage = 0
        self._item_memory_usage = {}
        self._cache = OrderedDict()
        super().__init__(*args, **kwargs)

    def first_key(self):
        if self._cache:
            return next(iter(self._cache))
        raise KeyError("Empty dictionary")

    def last_key(self):
        if self._cache:
            return next(reversed(self._cache))
        raise KeyError("Empty dictionary")

    def set(self, key, value, memory_usage=None):
        if key in self._cache:
            # Subtract old item memory usage if key already exists
            self.memory_usage -= self._item_memory_usage[key]

        self._cache[key] = value

        # Calculate and store new item memory usage
        item_memory_usage = max(get_model_size(value), memory_usage)
        self._item_memory_usage[key] = item_memory_usage
        self.memory_usage += item_memory_usage

    def pop(self, key):
        # Subtract item memory usage before deletion
        self.memory_usage -= self._item_memory_usage[key]
        del self._item_memory_usage[key]
        return self._cache.pop(key)

    def move_to_end(self, key, last=True):
        self._cache.move_to_end(key, last=last)

    def __contains__(self, item):
        return item in self._cache

    def __delitem__(self, key):
        self.pop(key)

    def __getitem__(self, item):
        return self._cache[item]

    def get(self, item):
        return self._cache.get(item)

    def __len__(self):
        return len(self._cache)

    def __bool__(self):
        return bool(self._cache)

    def keys(self):
        return self._cache.keys()


def get_mem_free_total(device):
    import psutil
    import torch

    if device.type == "cuda":
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        stats = torch.cuda.memory_stats(device)
        mem_active = stats["active_bytes.all.current"]
        mem_reserved = stats["reserved_bytes.all.current"]
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        mem_free_total *= 0.9
    else:
        # if we don't add a buffer, larger images come out as noise
        mem_free_total = psutil.virtual_memory().available * 0.6

    return mem_free_total


class GPUModelCache:
    def __init__(self, max_cpu_memory_gb="80%", max_gpu_memory_gb="95%", device=None):
        self._device = device
        if device in ("cpu", "mps"):
            # the "gpu" cache will be the only thing we use since there aren't two different memory stores in this case
            max_cpu_memory_gb = 0

        self._max_cpu_memory_gb = max_cpu_memory_gb
        self._max_gpu_memory_gb = max_gpu_memory_gb
        self.gpu_cache = MemoryTrackingCache()
        self.cpu_cache = MemoryTrackingCache()

    def stats_msg(self):
        import psutil

        msg = (
            f"    GPU cache: {len(self.gpu_cache)} items; {self.gpu_cache.memory_usage / (1024 ** 2):.1f} MB; Max: {self.max_gpu_memory / (1024 ** 2):.1f} MB;\n"
            f"    CPU cache: {len(self.cpu_cache)} items; {self.cpu_cache.memory_usage / (1024 ** 2):.1f} MB; Max: {self.max_cpu_memory / (1024 ** 2):.1f} MB;\n"
            f"    mem_free_total: {get_mem_free_total(self.device) / (1024 ** 2):.1f} MB; Ram Free: {psutil.virtual_memory().available / (1024 ** 2):.1f} MB;"
        )
        return msg

    @cached_property
    def device(self):
        import torch

        if self._device is None:
            self._device = get_device()

        if self._device in ("cpu", "mps", "mps:0"):
            # the "gpu" cache will be the only thing we use since there aren't two different memory stores in this case
            self._max_cpu_memory_gb = 0

        return torch.device(self._device)

    def make_gpu_space(self, bytes_to_free):
        import gc

        import torch.cuda

        log(self.stats_msg())
        log(f"Ensuring {bytes_to_free / (1024 ** 2):.1f} MB of GPU space.")

        while self.gpu_cache and (
            self.gpu_cache.memory_usage + bytes_to_free > self.max_gpu_memory
            or self.gpu_cache.memory_usage + bytes_to_free
            > get_mem_free_total(self.device)
        ):
            oldest_gpu_key = self.gpu_cache.first_key()
            self._move_to_cpu(oldest_gpu_key)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        if (
            self.gpu_cache.memory_usage + bytes_to_free > self.max_gpu_memory
            or self.gpu_cache.memory_usage + bytes_to_free
            > get_mem_free_total(self.device)
        ):
            msg = f"Unable to make {bytes_to_free / (1024 ** 2):.1f} MB space on {self.device}. \n{self.stats_msg()}"
            raise RuntimeError(msg)

    def make_cpu_space(self, bytes_to_free):
        import gc

        import psutil

        log(self.stats_msg())
        log(f"Ensuring {bytes_to_free / (1024 ** 2):.1f} MB of RAM space.")
        while self.cpu_cache and (
            self.cpu_cache.memory_usage + bytes_to_free > self.max_gpu_memory
            or self.cpu_cache.memory_usage + bytes_to_free
            > psutil.virtual_memory().available * 0.8
        ):
            oldest_cpu_key = self.cpu_cache.first_key()
            log(f"dropping {oldest_cpu_key} from memory")
            self.cpu_cache.pop(oldest_cpu_key)
            log(self.stats_msg())

        gc.collect()

    @cached_property
    def max_cpu_memory(self):
        _ = self.device
        if isinstance(self._max_cpu_memory_gb, str):
            if self._max_cpu_memory_gb.endswith("%"):
                import psutil

                total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
                pct_to_use = float(self._max_cpu_memory_gb[:-1]) / 100.0
                return total_ram_gb * pct_to_use * (1024**3)
            msg = f"Invalid value for max_cpu_memory_gb: {self._max_cpu_memory_gb}"
            raise ValueError(msg)
        return self._max_cpu_memory_gb * (1024**3)

    @cached_property
    def max_gpu_memory(self):
        _ = self.device
        if isinstance(self._max_gpu_memory_gb, str):
            if self._max_gpu_memory_gb.endswith("%"):
                import torch

                if self.device.type == "cuda":
                    device_props = torch.cuda.get_device_properties(0)
                    total_ram_gb = device_props.total_memory / (1024**3)
                else:
                    import psutil

                    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
                pct_to_use = float(self._max_gpu_memory_gb[:-1]) / 100.0
                return total_ram_gb * pct_to_use * (1024**3)
            msg = f"Invalid value for max_gpu_memory_gb: {self._max_gpu_memory_gb}"
            raise ValueError(msg)
        return self._max_gpu_memory_gb * (1024**3)

    def _move_to_gpu(self, key, model):
        model_size = get_model_size(model)
        if self.gpu_cache.memory_usage + model_size > self.max_gpu_memory:
            if len(self.gpu_cache) == 0:
                msg = f"GPU cache maximum ({self.max_gpu_memory / (1024 ** 2)} MB) is smaller than the item being cached ({model_size / 1024 ** 2} MB)."
                raise RuntimeError(msg)
            self.make_gpu_space(model_size)

        try:
            model_size = max(self.cpu_cache._item_memory_usage[key], model_size)
            self.cpu_cache.pop(key)
            log(f"dropping {key} from cpu cache")
        except KeyError:
            pass
        log(f"moving {key} to gpu")
        move_model_device(model, self.device)

        self.gpu_cache.set(key, value=model, memory_usage=model_size)

    def _move_to_cpu(self, key):
        import gc

        import psutil
        import torch

        memory_usage = self.gpu_cache._item_memory_usage[key]
        model = self.gpu_cache.pop(key)

        model_size = max(get_model_size(model), memory_usage)
        self.make_cpu_space(model_size)

        if (
            self.cpu_cache.memory_usage + model_size < self.max_cpu_memory
            and self.cpu_cache.memory_usage + model_size
            < psutil.virtual_memory().available * 0.8
        ):
            log(f"moving {key} to cpu")
            move_model_device(model, torch.device("cpu"))
            log(self.stats_msg())

            self.cpu_cache.set(key, model, memory_usage=model_size)
        else:
            log(f"dropping {key} from memory")
            del model
            gc.collect()
            log(self.stats_msg())

    def get(self, key):
        import torch

        if key not in self:
            msg = f"The key {key} does not exist in the cache"
            raise KeyError(msg)

        if key in self.cpu_cache and self.device != torch.device("cpu"):
            self.cpu_cache.move_to_end(key)
            self._move_to_gpu(key, self.cpu_cache[key])

        if key in self.gpu_cache:
            self.gpu_cache.move_to_end(key)

        model = self.gpu_cache.get(key)

        return model

    def __getitem__(self, key):
        return self.get(key)

    def set(self, key, model, memory_usage=0):
        from torch import nn

        if (
            hasattr(model, "model") and isinstance(model.model, nn.Module)
        ) or isinstance(model, nn.Module):
            pass
        else:
            raise ValueError("Only nn.Module objects can be cached")

        model_size = max(get_model_size(model), memory_usage)
        self.make_gpu_space(model_size)
        self._move_to_gpu(key, model)

    def __contains__(self, key):
        return key in self.gpu_cache or key in self.cpu_cache

    def keys(self):
        return list(self.cpu_cache.keys()) + list(self.gpu_cache.keys())

    def stats(self):
        return {
            "cpu_cache_count": len(self.cpu_cache),
            "cpu_cache_memory_usage": self.cpu_cache.memory_usage,
            "cpu_cache_max_memory": self.max_cpu_memory,
            "gpu_cache_count": len(self.gpu_cache),
            "gpu_cache_memory_usage": self.gpu_cache.memory_usage,
            "gpu_cache_max_memory": self.max_gpu_memory,
        }


class MemoryManagedModelWrapper:
    _mmmw_cache = GPUModelCache()

    def __init__(self, fn, namespace, estimated_ram_size_mb, *args, **kwargs):
        self._mmmw_fn = fn
        self._mmmw_args = args
        self._mmmw_kwargs = kwargs
        self._mmmw_namespace = namespace
        self._mmmw_estimated_ram_size_mb = estimated_ram_size_mb
        self._mmmw_cache_key = (namespace, *args, *tuple(kwargs.items()))

    def _mmmw_load_model(self):
        if self._mmmw_cache_key not in self.__class__._mmmw_cache:
            log(f"Loading model: {self._mmmw_cache_key}")
            self.__class__._mmmw_cache.make_gpu_space(
                self._mmmw_estimated_ram_size_mb * 1024**2
            )
            free_before = get_mem_free_total(self.__class__._mmmw_cache.device)
            model = self._mmmw_fn(*self._mmmw_args, **self._mmmw_kwargs)
            move_model_device(model, self.__class__._mmmw_cache.device)
            free_after = get_mem_free_total(self.__class__._mmmw_cache.device)
            log(f"Model loaded: {self._mmmw_cache_key} Used {free_after - free_before}")
            self.__class__._mmmw_cache.set(
                self._mmmw_cache_key,
                model,
                memory_usage=self._mmmw_estimated_ram_size_mb * 1024**2,
            )

        model = self.__class__._mmmw_cache[self._mmmw_cache_key]
        return model

    def __getattr__(self, name):
        model = self._mmmw_load_model()
        return getattr(model, name)

    def __call__(self, *args, **kwargs):
        model = self._mmmw_load_model()
        return model(*args, **kwargs)


def memory_managed_model(namespace, memory_usage_mb=0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return MemoryManagedModelWrapper(
                fn, namespace, memory_usage_mb, *args, **kwargs
            )

        return wrapper

    return decorator

import pytest
from torch import nn

from imaginairy.api import imagine
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device
from imaginairy.utils.model_cache import GPUModelCache


class DummyMemoryModule(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.large_layer = nn.Linear(in_features - 1, 1)

    def forward(self, x):
        return self.large_layer(x)


def create_model_of_n_bytes(n):
    import math

    n = int(math.floor(n / 4))
    return DummyMemoryModule(n)


@pytest.mark.skip()
@pytest.mark.parametrize(
    "model_version",
    [
        "SD-1.5",
        "openjourney-v1",
        "openjourney-v2",
        "openjourney-v4",
    ],
)
def test_memory_usage(filename_base_for_orig_outputs, model_version):
    """Test that we can switch between model versions."""
    prompt_text = "valley, fairytale treehouse village covered, , matte painting, highly detailed, dynamic lighting, cinematic, realism, realistic, photo real, sunset, detailed, high contrast, denoised, centered, michael whelan"
    prompts = [
        ImaginePrompt(prompt_text, model_weights=model_version, seed=1, steps=30)
    ]

    for i, result in enumerate(imagine(prompts)):
        img_path = f"{filename_base_for_orig_outputs}_{result.prompt.prompt_text}_{result.prompt.model}.png"
        result.img.save(img_path)


def test_get_nonexistent():
    cache = GPUModelCache(max_cpu_memory_gb=1, max_gpu_memory_gb=1)
    with pytest.raises(KeyError):
        cache.get("nonexistent_key")


@pytest.mark.skipif(get_device() == "cpu", reason="GPU not available")
def test_set_cpu_full():
    cache = GPUModelCache(
        max_cpu_memory_gb=0.000000001, max_gpu_memory_gb=0.01, device=get_device()
    )

    for i in range(4):
        cache.set(f"key{i}", create_model_of_n_bytes(4_000_000))
    assert len(cache.cpu_cache) == 0
    assert len(cache.gpu_cache) == 2


@pytest.mark.skipif(get_device() == "cpu", reason="GPU not available")
def test_set_gpu_full():
    device = get_device()
    cache = GPUModelCache(
        max_cpu_memory_gb=1, max_gpu_memory_gb=0.0000001, device=device
    )
    if device in ("cpu", "mps"):
        assert cache.max_cpu_memory == 0
    else:
        assert cache.max_cpu_memory == 1073741824
    model = create_model_of_n_bytes(100_000)
    with pytest.raises(RuntimeError):
        cache.set("key1", model)


@pytest.mark.skipif(get_device() == "cpu", reason="GPU not available")
def test_get_existing_cpu():
    cache = GPUModelCache(max_cpu_memory_gb=0.1, max_gpu_memory_gb=0.1, device="cpu")
    model = create_model_of_n_bytes(10_000)
    cache.set("key", model)
    retrieved_data = cache.get("key")
    assert retrieved_data == model
    # assert 'key' in cache.cpu_cache
    assert "key" in cache.gpu_cache


@pytest.mark.skipif(get_device() == "cpu", reason="GPU not available")
def test_get_existing_move_to_gpu():
    cache = GPUModelCache(
        max_cpu_memory_gb=0.1, max_gpu_memory_gb=0.1, device=get_device()
    )
    model = create_model_of_n_bytes(10_000)
    cache.set("key", model)
    retrieved_data = cache.get("key")
    assert retrieved_data == model
    assert "key" not in cache.cpu_cache
    assert "key" in cache.gpu_cache


@pytest.mark.skipif(get_device() == "cpu", reason="GPU not available")
def test_cache_ordering():
    cache = GPUModelCache(
        max_cpu_memory_gb=0.01, max_gpu_memory_gb=0.01, device=get_device()
    )

    cache.set("key-0", create_model_of_n_bytes(4_000_000))
    assert list(cache.cpu_cache.keys()) == []
    assert list(cache.gpu_cache.keys()) == ["key-0"]
    assert (cache.cpu_cache.memory_usage, cache.gpu_cache.memory_usage) == (
        0,
        4_000_000,
    )

    cache.set("key-1", create_model_of_n_bytes(4_000_000))
    assert list(cache.cpu_cache.keys()) == []
    assert list(cache.gpu_cache.keys()) == ["key-0", "key-1"]
    assert (cache.cpu_cache.memory_usage, cache.gpu_cache.memory_usage) == (
        0,
        8_000_000,
    )

    cache.set("key-2", create_model_of_n_bytes(4_000_000))
    assert list(cache.cpu_cache.keys()) == ["key-0"]
    assert list(cache.gpu_cache.keys()) == ["key-1", "key-2"]
    assert (cache.cpu_cache.memory_usage, cache.gpu_cache.memory_usage) == (
        4_000_000,
        8_000_000,
    )

    cache.set("key-3", create_model_of_n_bytes(4_000_000))
    assert list(cache.cpu_cache.keys()) == ["key-0", "key-1"]
    assert list(cache.gpu_cache.keys()) == ["key-2", "key-3"]
    assert (cache.cpu_cache.memory_usage, cache.gpu_cache.memory_usage) == (
        8_000_000,
        8_000_000,
    )

    cache.set("key-4", create_model_of_n_bytes(4_000_000))
    assert list(cache.cpu_cache.keys()) == ["key-1", "key-2"]
    assert list(cache.gpu_cache.keys()) == ["key-3", "key-4"]
    assert list(cache.keys()) == ["key-1", "key-2", "key-3", "key-4"]
    assert (cache.cpu_cache.memory_usage, cache.gpu_cache.memory_usage) == (
        8_000_000,
        8_000_000,
    )

    cache.get("key-2")
    assert list(cache.keys()) == ["key-3", "key-4", "key-2"]

    cache.set("key-5", create_model_of_n_bytes(9_000_000))
    assert list(cache.cpu_cache.keys()) == ["key-4", "key-2"]
    assert list(cache.gpu_cache.keys()) == ["key-5"]
    assert list(cache.keys()) == ["key-4", "key-2", "key-5"]

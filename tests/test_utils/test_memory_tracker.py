import pytest

from imaginairy.utils.memory_tracker import TorchRAMTracker


class MockedMemory:
    allocated_memory = 0
    peak_memory = 0

    @classmethod
    def allocate_memory(cls, amount):
        cls.allocated_memory += amount
        cls.peak_memory = max(cls.peak_memory, cls.allocated_memory)

    @classmethod
    def free_memory(cls, amount):
        cls.allocated_memory = max(cls.allocated_memory - amount, 0)

    @classmethod
    def memory_allocated(cls):
        return cls.allocated_memory

    @classmethod
    def max_memory_allocated(cls):
        return cls.peak_memory

    @classmethod
    def reset_peak_memory_stats(cls):
        cls.peak_memory = cls.allocated_memory


@pytest.fixture()
def mocked_memory(monkeypatch):
    monkeypatch.setattr(TorchRAMTracker, "mem_interface", MockedMemory)
    MockedMemory.allocated_memory = 0
    MockedMemory.peak_memory = 0
    return MockedMemory


def test_torch_ram_tracker_basic(mocked_memory):
    with TorchRAMTracker("a") as trt_a:
        mocked_memory.allocate_memory(1000)
        mocked_memory.free_memory(1000)

    assert trt_a.peak_memory == 1000


def test_torch_ram_tracker_basic_cumulative(mocked_memory):
    with TorchRAMTracker("a") as trt_a:
        mocked_memory.allocate_memory(500)
        mocked_memory.free_memory(100)
        mocked_memory.allocate_memory(500)

    assert trt_a.peak_memory == 900


def test_torch_ram_tracker_nested(mocked_memory):
    with TorchRAMTracker("a") as trt_a:
        mocked_memory.allocate_memory(1000)
        mocked_memory.free_memory(1000)
        with TorchRAMTracker("b") as trt_b:
            mocked_memory.allocate_memory(100)
            mocked_memory.free_memory(100)

    assert trt_a.peak_memory == 1000
    assert trt_b.peak_memory == 100


def test_torch_ram_tracker_nested_b(mocked_memory):
    with TorchRAMTracker("a") as trt_a:
        mocked_memory.allocate_memory(100)
        mocked_memory.free_memory(100)
        with TorchRAMTracker("b") as trt_b:
            mocked_memory.allocate_memory(1000)
            mocked_memory.free_memory(1000)

    assert trt_a.peak_memory == 1000
    assert trt_b.peak_memory == 1000


def test_torch_ram_tracker_nested_deep(mocked_memory):
    with TorchRAMTracker("a") as trt_a:
        mocked_memory.allocate_memory(10000)
        with TorchRAMTracker("b") as trt_b:
            mocked_memory.free_memory(1000)
            with TorchRAMTracker("c") as trt_c:
                mocked_memory.free_memory(1000)
                with TorchRAMTracker("d") as trt_d:
                    mocked_memory.free_memory(1000)
                    with TorchRAMTracker("e") as trt_e:
                        mocked_memory.free_memory(1000)
                        with TorchRAMTracker("f") as trt_f:
                            mocked_memory.free_memory(1000)

    assert trt_a.peak_memory == 10000
    assert trt_b.peak_memory == 10000
    assert trt_c.peak_memory == 9000
    assert trt_d.peak_memory == 8000
    assert trt_e.peak_memory == 7000
    assert trt_f.peak_memory == 6000


def test_torch_ram_tracker(mocked_memory):
    with TorchRAMTracker("a") as trt_a:
        mocked_memory.allocate_memory(1000)  # Spike in block A
        mocked_memory.free_memory(900)
        with TorchRAMTracker("b") as trt_b:
            mocked_memory.allocate_memory(50)  # Operations in block B
            mocked_memory.free_memory(25)
        mocked_memory.free_memory(75)
        mocked_memory.allocate_memory(30)  # More operations in block A/C

    with TorchRAMTracker("c") as trt_c:
        mocked_memory.allocate_memory(80)  # Operations in another block
        with TorchRAMTracker("d") as trt_d:
            mocked_memory.allocate_memory(40)
        mocked_memory.free_memory(60)
        mocked_memory.allocate_memory(600)

    assert trt_a.peak_memory == 1000
    assert trt_b.peak_memory == 150
    assert trt_c.peak_memory == 740
    assert trt_d.peak_memory == 200

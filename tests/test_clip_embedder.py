import hashlib

import pytest
from safetensors import safe_open

from imaginairy.modules.clip_embedders import FrozenCLIPEmbedder
from imaginairy.utils import get_device
from tests import TESTS_FOLDER


def hash_tensor(t):
    t = t.cpu().detach().numpy().tobytes()
    return hashlib.md5(t).hexdigest()


@pytest.mark.skipif(get_device() == "cpu", reason="Too slow to run on CPU")
def test_text_conditioning():
    embedder = FrozenCLIPEmbedder()
    embedder.to(get_device())
    neutral_embedding = embedder.encode([""]).to("cpu")
    with safe_open(
        f"{TESTS_FOLDER}/data/neutral_clip_embedding_mps.safetensors",
        framework="pt",
        device="cpu",
    ) as f:
        neutral_embedding_mps_expected = f.get_tensor("neutral_clip_embedding_mps")

    diff = neutral_embedding - neutral_embedding_mps_expected
    assert diff.sum() < 0.09

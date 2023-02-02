from imaginairy import config
from imaginairy.samplers import SAMPLER_TYPE_OPTIONS


def test_sampler_options():
    assert set(config.SAMPLER_TYPE_OPTIONS) == set(SAMPLER_TYPE_OPTIONS)

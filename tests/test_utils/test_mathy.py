from imaginairy.utils.mathy import make_odd


def test_make_odd():
    assert make_odd(0) == 1
    assert make_odd(1) == 1
    assert make_odd(2) == 3
    assert make_odd(3) == 3
    assert make_odd(4) == 5
    assert make_odd(4.1) == 5
    assert make_odd(-1) == -1
    assert make_odd(-2) == -1
    assert make_odd(1000) == 1001

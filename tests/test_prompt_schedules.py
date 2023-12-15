import pytest

from imaginairy.utils import frange
from imaginairy.utils.prompt_schedules import parse_schedule_str


@pytest.mark.parametrize(
    ("schedule_str", "expected"),
    [
        ("prompt_strength[2:40:1]", ("prompt_strength", list(range(2, 40)))),
        ("prompt_strength[2:40:0.5]", ("prompt_strength", list(frange(2, 40, 0.5)))),
        ("prompt_strength[2,5,10,15]", ("prompt_strength", [2, 5, 10, 15])),
        (
            "prompt_strength[red,blue,10,15]",
            ("prompt_strength", ["red", "blue", 10, 15]),
        ),
    ],
)
def test_parse_schedule_str(schedule_str, expected):
    cleaned_schedules = parse_schedule_str(schedule_str)
    assert cleaned_schedules == expected

from datetime import datetime, timezone

import pytest


def format_filename(format_template: str, data: dict) -> str:
    """
    Formats the filename based on the provided template and variables.
    """
    if not isinstance(format_template, str):
        raise TypeError("format argument must be a string")

    filename = format_template.format(**data)
    filename += data["ext"]
    return filename


base_data = {
    "original": "file",
    "number": 1,
    "algorithm": "alg",
    "now": datetime(2023, 1, 23, 12, 30, 45, tzinfo=timezone.utc),
    "ext": ".jpg",
}


@pytest.mark.parametrize(
    ("format_str", "data", "expected"),
    [
        ("{original}_{algorithm}", base_data, "file_alg.jpg"),
        (
            "{original}_{number}_{now}",
            base_data,
            "file_1_2023-01-23 12:30:45+00:00.jpg",
        ),
        ("", base_data, ".jpg"),
        ("{original}", {}, KeyError),
        ("{nonexistent_key}", base_data, KeyError),
        (123, base_data, TypeError),
        ("{original}_@#$_{algorithm}", base_data, "file_@#$_alg.jpg"),
        ("{original}" * 100, base_data, "file" * 100 + ".jpg"),
        (
            "{original}_{number}",
            {"original": "file", "number": 123, "ext": ".jpg"},
            "file_123.jpg",
        ),
        (
            "{now}",
            {"now": "2023/01/23", "ext": ".log"},
            "2023/01/23.log",
        ),
        ("{original}", {"original": "file", "ext": ""}, "file"),
    ],
)
def test_format_filename(format_str, data, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        try:
            format_filename(format_str, data)
        except expected:
            assert True, f"Expected {expected} to be raised"
        except Exception:
            raise
    else:
        assert format_filename(format_str, data) == expected

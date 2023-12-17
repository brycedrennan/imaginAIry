import pyparsing as pp
import pytest
from pyparsing import ParseException

from imaginairy.utils.spaced_kv_parser import parse_spaced_key_value_pairs


def test_basic_parsing():
    input_str = "text='Hello World' font='Arial' size=12 color='#FF0000'"

    expected = {
        "text": "Hello World",
        "font": "Arial",
        "size": "12",
        "color": "#FF0000",
    }

    try:
        assert parse_spaced_key_value_pairs(input_str) == expected
    except pp.ParseException as e:
        print(e.explain())
        raise


def test_unquoted_values():
    input_str = "width=800 height=600 bg_color=#FFFFFF"
    expected = {"width": "800", "height": "600", "bg_color": "#FFFFFF"}
    assert parse_spaced_key_value_pairs(input_str) == expected


def test_mixed_quoted_unquoted():
    input_str = "title='My Title' resolution=1080p"
    expected = {"title": "My Title", "resolution": "1080p"}
    assert parse_spaced_key_value_pairs(input_str) == expected


def test_empty_string():
    input_str = ""
    expected = {}
    assert parse_spaced_key_value_pairs(input_str) == expected


def test_invalid_format():
    input_str = "This is not a valid format"
    with pytest.raises(ParseException):  # noqa
        parse_spaced_key_value_pairs(input_str)


def test_only_keys():
    input_str = "key1= key2="
    expected = {"key1": "", "key2": ""}
    assert parse_spaced_key_value_pairs(input_str) == expected


challenging_test_queries = [
    ("foo=\"bar'baz\" bar='foo\"bar'", {"foo": "bar'baz", "bar": 'foo"bar'}),
    ("foo=\"'bar'\" bar='\"baz\"'", {"foo": "'bar'", "bar": '"baz"'}),
    ("foo=\"bar\\\"baz\" bar='foo\\'bar'", {"foo": 'bar\\"baz', "bar": "foo\\'bar"}),
    (
        'special=👍 emoji="😀 😃" text="This is a test\\nwith newline"',
        {"special": "👍", "emoji": "😀 😃", "text": "This is a test\\nwith newline"},
    ),
    ('name=" John  Doe " age=" 30 "', {"name": " John  Doe ", "age": " 30 "}),
    ("special=@@!!", {"special": "@@!!"}),
    (
        'text="This is a test\\\\nwith incomplete escape\\\\"',
        {"text": "This is a test\\\\nwith incomplete escape\\\\"},
    ),
    ("foo= bar=", {"foo": "", "bar": ""}),
    ("foo= bar=30.4 zab=-1.2", {"foo": "", "bar": "30.4", "zab": "-1.2"}),
    ("", {}),
    (
        'foo="bar" baz=\'qux\' specialChars="@@!!" empty= complex="\'This is a \\"complex\\" string\'"',
        {
            "foo": "bar",
            "baz": "qux",
            "specialChars": "@@!!",
            "empty": "",
            "complex": "'This is a \\\"complex\\\" string'",
        },
    ),
]


@pytest.mark.parametrize(("query", "expected"), challenging_test_queries)
def test_challenging_queries(query, expected):
    data = parse_spaced_key_value_pairs(query)
    assert data == expected

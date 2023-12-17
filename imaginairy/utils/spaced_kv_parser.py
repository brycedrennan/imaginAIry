from functools import lru_cache

from pyparsing import (
    CharsNotIn,
    Group,
    OneOrMore,
    Optional,
    Word,
    alphanums,
    alphas,
    quotedString,
    removeQuotes,
)


@lru_cache
def _make_attribute_parser():
    key_parser = Word(alphas, alphanums + "_")
    quoted_value_parser = quotedString.setParseAction(removeQuotes)
    unquoted_value_parser = CharsNotIn(" =\"'")
    key_value_pair_parser = (
        key_parser
        + "="
        + Optional(quoted_value_parser | unquoted_value_parser, default="")
    )
    multiple_pairs_parser = OneOrMore(Group(key_value_pair_parser))
    return multiple_pairs_parser


def parse_spaced_key_value_pairs(text: str) -> dict[str, str]:
    """
    Parses a string of key-value pairs separated by spaces.

    :param text: String of key-value pairs separated by spaces.
    :return: List of key-value pairs.
    """
    if not text:
        return {}

    rows = _make_attribute_parser().parseString(text, parseAll=True)
    data = {r[0]: r[2] for r in rows}
    return data


def parse_spaced_key_value_pairs_html(text: str):
    html_version = f"<foo {text}>"
    parsed_html = parse_html_tag(html_version)
    return parsed_html["attributes"]


def parse_html_tag(html_tag):
    """
    Parses a single HTML tag and returns a dictionary with the tag name and its attributes.

    Args:
    html_tag (str): A string representing the HTML tag to be parsed.

    Returns:
    dict: A dictionary with 'tagname' and 'attributes'. 'tagname' is a string and 'attributes' is a dictionary.
    """

    from html.parser import HTMLParser

    class MyHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tagname = ""
            self.attributes = {}

        def handle_starttag(self, tag, attrs):
            self.tagname = tag
            self.attributes = dict(attrs)

    parser = MyHTMLParser()
    parser.feed(html_tag)
    return {"tagname": parser.tagname, "attributes": parser.attributes}


def parse_spaced_key_value_pairs_re(text: str) -> dict[str, str]:
    """
    Parses a string of key-value pairs separated by spaces.

    :param text: String of key-value pairs separated by spaces.
    :return: List of key-value pairs.
    """
    if not text:
        return {}
    import re

    # Building regex parts for readability
    key_pattern = r"(?P<key>\w+)"
    quoted_value_pattern = r'(?:"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')'
    unquoted_value_pattern = r'[^\'"\s]*'
    value_pattern = f"(?P<value>{quoted_value_pattern}|{unquoted_value_pattern})"

    # Complete pattern with named groups
    pattern = rf"{key_pattern}={value_pattern}"

    # Find all matches
    matches = re.findall(pattern, text)

    # Validate the query string format
    if not matches and text:
        raise ValueError("Invalid format")

    parsed_query = {}
    for key, value in matches:
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            # Remove quotes and handle escape sequences
            value = bytes(value[1:-1], "utf-8").decode("unicode_escape")
        parsed_query[key] = value

    return parsed_query

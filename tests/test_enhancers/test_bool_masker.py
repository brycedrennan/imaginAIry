import pytest

from imaginairy.enhancers.bool_masker import MASK_PROMPT

boolean_mask_test_cases = [
    (
        "fruit bowl",
        "'fruit bowl'",
    ),
    (
        "((((fruit bowl))))",
        "'fruit bowl'",
    ),
    (
        "fruit OR bowl",
        "('fruit' OR 'bowl')",
    ),
    (
        "fruit|bowl",
        "('fruit' OR 'bowl')",
    ),
    (
        "fruit | bowl",
        "('fruit' OR 'bowl')",
    ),
    (
        "fruit OR bowl OR pear",
        "('fruit' OR 'bowl' OR 'pear')",
    ),
    (
        "fruit AND bowl",
        "('fruit' AND 'bowl')",
    ),
    (
        "fruit & bowl",
        "('fruit' AND 'bowl')",
    ),
    (
        "fruit AND NOT green",
        "('fruit' AND NOT 'green')",
    ),
    (
        "fruit bowl{+0.5}",
        "'fruit bowl'+0.5",
    ),
    (
        "fruit bowl{+0.5} OR fruit",
        "('fruit bowl'+0.5 OR 'fruit')",
    ),
    (
        "NOT pizza",
        "NOT 'pizza'",
    ),
    (
        "car AND (wheels OR trunk OR engine OR windows) AND NOT (truck OR headlights{*10})",
        "('car' AND ('wheels' OR 'trunk' OR 'engine' OR 'windows') AND NOT ('truck' OR 'headlights'*10))",
    ),
    (
        "car AND (wheels OR trunk OR engine OR windows OR headlights) AND NOT (truck OR headlights){*10}",
        "('car' AND ('wheels' OR 'trunk' OR 'engine' OR 'windows' OR 'headlights') AND NOT ('truck' OR 'headlights')*10)",
    ),
]


@pytest.mark.parametrize(("mask_text", "expected"), boolean_mask_test_cases)
def test_clip_mask_parser(mask_text, expected):
    parsed = MASK_PROMPT.parseString(mask_text)[0][0]
    assert str(parsed) == expected

# pylama:ignore=W0613
"""
Logic for parsing mask prompts.

Supports
 lower case text descriptions
 Combinations: AND OR NOT ()
 Strength Modifiers: {<operator><number>}

Examples:
  fruit
  fruit bowl
  fruit AND NOT pears
  fruit OR bowl
  (pears OR oranges OR peaches){*1.5}
  fruit{-0.1} OR bowl

"""
import operator
from abc import ABC

import pyparsing as pp
import torch
from pyparsing import ParserElement

ParserElement.enablePackrat()


class Mask(ABC):
    def get_mask_for_image(self, img):
        pass

    def gather_text_descriptions(self):
        return set()

    def apply_masks(self, mask_cache):
        pass


class SimpleMask(Mask):
    def __init__(self, text):
        self.text = text

    @classmethod
    def from_simple_prompt(cls, instring, tokens_start, ret_tokens):
        return cls(text=ret_tokens[0])

    def __repr__(self):
        return f"'{self.text}'"

    def gather_text_descriptions(self):
        return {self.text}

    def apply_masks(self, mask_cache):
        return mask_cache[self.text]


class ModifiedMask(Mask):
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        # '%': operator.mod,
        # '^': operator.xor,
    }

    def __init__(self, mask, modifier):
        if modifier:
            modifier = modifier.strip("{}")
        self.mask = mask
        self.modifier = modifier
        self.operand = self.ops[modifier[0]]
        self.value = float(modifier[1:])

    @classmethod
    def from_modifier_parse(cls, instring, tokens_start, ret_tokens):
        return cls(mask=ret_tokens[0][0], modifier=ret_tokens[0][1])

    def __repr__(self):
        return f"{repr(self.mask)}{self.modifier}"

    def gather_text_descriptions(self):
        return self.mask.gather_text_descriptions()

    def apply_masks(self, mask_cache):
        mask = self.mask.apply_masks(mask_cache)
        return torch.clamp(self.operand(mask, self.value), 0, 1)


class NestedMask(Mask):
    def __init__(self, masks, op):
        self.masks = masks
        self.op = op

    @classmethod
    def from_or(cls, instring, tokens_start, ret_tokens):
        sub_masks = [t for t in ret_tokens[0] if isinstance(t, Mask)]
        return cls(masks=sub_masks, op="OR")

    @classmethod
    def from_and(cls, instring, tokens_start, ret_tokens):
        sub_masks = [t for t in ret_tokens[0] if isinstance(t, Mask)]
        return cls(masks=sub_masks, op="AND")

    @classmethod
    def from_not(cls, instring, tokens_start, ret_tokens):
        sub_masks = [t for t in ret_tokens[0] if isinstance(t, Mask)]
        assert len(sub_masks) == 1
        return cls(masks=sub_masks, op="NOT")

    def __repr__(self):
        if self.op == "NOT":
            return f"NOT {self.masks[0]}"
        sub = f" {self.op} ".join(repr(m) for m in self.masks)
        return f"({sub})"

    def gather_text_descriptions(self):
        return set().union(*[m.gather_text_descriptions() for m in self.masks])

    def apply_masks(self, mask_cache):
        submasks = [m.apply_masks(mask_cache) for m in self.masks]
        mask = submasks[0]
        if self.op == "OR":
            for submask in submasks:
                mask = torch.maximum(mask, submask)
        elif self.op == "AND":
            for submask in submasks:
                mask = torch.minimum(mask, submask)
        elif self.op == "NOT":
            mask = 1 - mask
        else:
            raise ValueError(f"Invalid operand {self.op}")
        return torch.clamp(mask, 0, 1)


AND = (pp.Literal("AND") | pp.Literal("&")).setName("AND").setResultsName("op")
OR = (pp.Literal("OR") | pp.Literal("|")).setName("OR").setResultsName("op")
NOT = (pp.Literal("NOT") | pp.Literal("!")).setName("NOT").setResultsName("op")

PROMPT_MODIFIER = (
    pp.Regex(r"{[*/+-]\d+\.?\d*}")
    .setName("prompt_modifier")
    .setResultsName("prompt_modifier")
)
PROMPT_TEXT = (
    pp.Regex(r"[a-z0-9]?[a-z0-9 -]*[a-z0-9]")
    .setName("prompt_text")
    .setResultsName("prompt_text")
)
SIMPLE_PROMPT = PROMPT_TEXT.setResultsName("simplePrompt")
SIMPLE_PROMPT.setParseAction(SimpleMask.from_simple_prompt)

COMPLEX_PROMPT = pp.infixNotation(
    SIMPLE_PROMPT,
    [
        (PROMPT_MODIFIER, 1, pp.opAssoc.LEFT, ModifiedMask.from_modifier_parse),
        (NOT, 1, pp.opAssoc.RIGHT, NestedMask.from_not),
        (AND, 2, pp.opAssoc.LEFT, NestedMask.from_and),
        (OR, 2, pp.opAssoc.LEFT, NestedMask.from_or),
    ],
)
MASK_PROMPT = pp.Group(COMPLEX_PROMPT).setResultsName("complexPrompt")

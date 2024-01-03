import gzip
import re
from functools import lru_cache
from itertools import islice
from pathlib import Path

from torch import Tensor, tensor

import imaginairy.vendored.refiners.fluxion.layers as fl
from imaginairy.vendored.refiners.fluxion import pad


class CLIPTokenizer(fl.Module):
    def __init__(
        self,
        vocabulary_path: str | Path = Path(__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz",
        sequence_length: int = 77,
        start_of_text_token_id: int = 49406,
        end_of_text_token_id: int = 49407,
        pad_token_id: int = 49407,
    ) -> None:
        super().__init__()
        self.vocabulary_path = vocabulary_path
        self.sequence_length = sequence_length
        self.byte_to_unicode_mapping = self.get_bytes_to_unicode_mapping()
        self.byte_decoder = {v: k for k, v in self.byte_to_unicode_mapping.items()}
        merge_tuples = [
            tuple(merge.split())
            for merge in gzip.open(filename=vocabulary_path)
            .read()
            .decode(encoding="utf-8")
            .split(sep="\n")[1 : 49152 - 256 - 2 + 1]
        ]
        vocabulary = (
            list(self.byte_to_unicode_mapping.values())
            + [v + "</w>" for v in self.byte_to_unicode_mapping.values()]
            + ["".join(merge) for merge in merge_tuples]
            + ["", ""]
        )
        self.token_to_id_mapping = {token: i for i, token in enumerate(iterable=vocabulary)}
        self.byte_pair_encoding_ranks = {merge: i for i, merge in enumerate(iterable=merge_tuples)}
        self.byte_pair_encoding_cache = {"": ""}
        # Note: this regular expression does not support Unicode. It was changed so
        # to get rid of the dependence on the `regex` module. Unicode support could
        # potentially be added back by leveraging the `\w` character class.
        self.token_pattern = re.compile(
            pattern=r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s\w]+""",
            flags=re.IGNORECASE,
        )
        self.start_of_text_token_id: int = start_of_text_token_id
        self.end_of_text_token_id: int = end_of_text_token_id
        self.pad_token_id: int = pad_token_id

    def forward(self, text: str) -> Tensor:
        tokens = self.encode(text=text, max_length=self.sequence_length).unsqueeze(dim=0)
        assert (
            tokens.shape[1] <= self.sequence_length
        ), f"Text is too long: tokens.shape[1] > sequence_length: {tokens.shape[1]} > {self.sequence_length}"
        return pad(x=tokens, pad=(0, self.sequence_length - tokens.shape[1]), value=self.pad_token_id)

    @lru_cache()
    def get_bytes_to_unicode_mapping(self) -> dict[int, str]:
        initial_byte_values = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        extra_unicode_values = (byte for byte in range(2**8) if byte not in initial_byte_values)
        byte_values = initial_byte_values + list(extra_unicode_values)
        unicode_values = [chr(value) for value in byte_values]
        return dict(zip(byte_values, unicode_values))

    def byte_pair_encoding(self, token: str) -> str:
        if token in self.byte_pair_encoding_cache:
            return self.byte_pair_encoding_cache[token]

        def recursive_bpe(word: tuple[str, ...]) -> tuple[str, ...]:
            if len(word) < 2:
                return word
            pairs = {(i, (word[i], word[i + 1])) for i in range(len(word) - 1)}
            min_pair = min(
                pairs,
                key=lambda pair: self.byte_pair_encoding_ranks.get(pair[1], float("inf")),
            )
            if min_pair[1] not in self.byte_pair_encoding_ranks:
                return word
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i == min_pair[0]:
                    new_word.append(min_pair[1][0] + min_pair[1][1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            return recursive_bpe(tuple(new_word))

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        result = " ".join(recursive_bpe(word=word))
        self.byte_pair_encoding_cache[token] = result
        return result

    def encode(self, text: str, max_length: int | None = None) -> Tensor:
        text = re.sub(pattern=r"\s+", repl=" ", string=text.lower())
        tokens = re.findall(pattern=self.token_pattern, string=text)
        upper_bound = None
        if max_length:
            assert max_length >= 2
            upper_bound = max_length - 2
        encoded_tokens = islice(
            (
                self.token_to_id_mapping[subtoken]
                for token in tokens
                for subtoken in self.byte_pair_encoding(
                    token="".join(self.byte_to_unicode_mapping[character] for character in token.encode("utf-8"))
                ).split(sep=" ")
            ),
            0,
            upper_bound,
        )
        return tensor(data=[self.start_of_text_token_id, *encoded_tokens, self.end_of_text_token_id])

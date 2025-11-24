from __future__ import annotations
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Dict
import tiktoken
from tiktoken.load import load_tiktoken_bpe


@dataclass
class TikTokenizerEncoding:
    ids: list[int]


class TikTokenizerWrapper:
    num_reserved_special_tokens = 256
    pat_str = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    def __init__(
        self,
        vocab_path: str,
        added_tokens_decoder: Dict[str, Dict],
    ):
        mergeable_ranks = load_tiktoken_bpe(vocab_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens_mapping = {
            int(i): meta["content"] for i, meta in added_tokens_decoder.items()
        }
        self.special_tokens = {
            special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
            for i in range(
                num_base_tokens,
                num_base_tokens + self.num_reserved_special_tokens + 2,
            )
        }
        self.encoding = tiktoken.Encoding(
            name="tiktoken",
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.allowed_special = set(self.special_tokens.keys())
        self.model = SimpleNamespace(unk_token=None)

    def no_truncation(self):
        return

    def encode(self, text: str, add_special_tokens: bool = False):
        allowed = self.allowed_special if add_special_tokens else set()
        ids = self.encoding.encode(text, allowed_special=allowed, disallowed_special=())
        return TikTokenizerEncoding(ids)

    def decode(self, ids):
        return self.encoding.decode(ids)

    def token_to_id(self, token: str):
        try:
            return self.encoding.token_to_id(token)
        except KeyError:
            return None

    def id_to_token(self, idx: int):
        return self.encoding.id_to_token(idx)

    def get_vocab_size(self):
        return self.encoding.n_vocab

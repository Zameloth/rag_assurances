"""BGE-M3 token counting — the shared band measure (SPEC §4, ADR-0003).

Every chunker (articles here, fiches in #24) measures against the tokenizer BGE-M3 itself
embeds with, never a character or word approximation. Counting **includes** the `<s>`/
`</s>` special tokens the encoder adds by default — SPEC §4's own measured numbers (article
median 167, p90 582, 289 over 512) were taken this way, so excluding them would silently
drift the band off the numbers the spec documents.

Only the tokenizer is loaded here, not the model: chunking needs token counts, not
embeddings, and pulling in the model weights (#26, ~2.3 GB) for a text-processing step
would tie it to a multi-GB download it never uses.
"""

from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

__all__ = ["count_tokens", "count_content_tokens", "SPECIAL_TOKEN_COUNT"]

_MODEL_ID = "BAAI/bge-m3"

# `<s>` + `</s>` — constant regardless of input, and the whole reason `count_content_tokens`
# is additive across concatenated spans (verified empirically: encoding two spans separately
# without specials and summing equals encoding their space-joined concatenation as one).
SPECIAL_TOKEN_COUNT = 2


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(_MODEL_ID)


def count_tokens(text: str) -> int:
    """Token count under the BGE-M3 tokenizer, including the `<s>`/`</s>` special tokens."""
    return len(_tokenizer().encode(text).ids)


def count_content_tokens(text: str) -> int:
    """Token count without the wrapping specials.

    Additive across concatenated, space-joined spans — the packer sums this per block
    instead of re-tokenizing the whole accumulated chunk on every step, and adds
    `SPECIAL_TOKEN_COUNT` once at the end to get what `count_tokens` on the joined text
    would have produced.
    """
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)

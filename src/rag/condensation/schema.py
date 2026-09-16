"""The condenser's output contract (SPEC §8.3, ADR-0008, #43).

`{"requete": str}` — one field, not two. A `passthrough: bool` alongside the text was
rejected (ADR-0008): two fields can disagree with no principled winner, while string
equality against the raw turn (`condensed == raw`) is a perfect passthrough detector and
cannot contradict itself. `Reponse`/`Refus` need a container for `with_structured_output`
because their union puts an `anyOf` at the schema root, which `strict: true` rejects
(`rag.generation.chain._EnvelopeContainer`); `CondenserOutput` is already a plain object at
the root, so no such wrapper is needed here.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["CondenserOutput"]


class CondenserOutput(BaseModel):
    """`requete` is the condenser's one job: a standalone French question, or the raw turn
    returned verbatim when it already was one (SPEC §8.6) — the sanitizer
    (`rag.condensation.sanitizer`), not this schema, is what actually enforces that shape."""

    requete: str

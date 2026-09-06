"""Shared candidate builders for `rag.generation` tests (SPEC §10, #42) — one fiche,
article and no-article-marker shape, factored out of `test_generation_citation.py`,
`test_generation_pipeline.py` and `test_generation_prompt.py`.

Plain functions, called directly, not pytest fixtures: a `conftest.py` under
`tests/generation` would collide on module name with `tests/conftest.py` under
`mypy --strict`'s flat, `__init__.py`-free `tests` tree
(`pyproject.toml`'s `files = ["src", "tests", "scripts"]`), and importing
fixture-decorated functions by name into a test module only to re-declare the same name
as a parameter trips `ruff`'s F811 (redefinition of an unused import).
"""

from __future__ import annotations

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID, NO_ARTICLE_MARKER_TEXT

__all__ = ["make_article", "make_fiche", "no_article_marker"]


def make_fiche(**payload: object) -> Candidate:
    return Candidate(
        id="fiche-1",
        score=0.9,
        register=Register.FICHE,
        payload=payload,
        provenance=frozenset({Provenance.SEARCH}),
    )


def make_article(citation_id: str, **extra: object) -> Candidate:
    payload: dict[str, object] = {"citation_id": citation_id, "text": "...", **extra}
    return Candidate(
        id=f"point-{citation_id}",
        score=0.9,
        register=Register.ARTICLE,
        payload=payload,
        provenance=frozenset({Provenance.SEARCH}),
    )


def no_article_marker() -> Candidate:
    return Candidate(
        id=NO_ARTICLE_MARKER_ID,
        score=0.0,
        register=Register.ARTICLE,
        payload={"text": NO_ARTICLE_MARKER_TEXT},
        provenance=frozenset(),
    )

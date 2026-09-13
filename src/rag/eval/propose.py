"""Pure helpers behind the annotation helper's AI-assisted proposals (#70, ADR-0020).

Factored out of `annotate_app.py` so the two structural guarantees ADR-0020 requires are
unit-testable without a FastAPI app, a corpus fixture, or a network call:

- `lexical_shortlist` is the **entire** `gold_articles` shortlist algorithm on the corpus
  side of the LLM call — hand-rolled term overlap, no embedding model, no import of
  `rag.retrieval` — which is what keeps a `gold_articles` proposal independent of the
  pipeline under test (ADR-0020's central constraint).
- `filter_verbatim_spans` is the server-side enforcement of SPEC §12.2's verbatim
  invariant for `gold_spans` proposals: a candidate quote the LLM invented or mis-copied
  is dropped here, structurally, rather than trusted.

Neither function calls an LLM or touches disk; `annotate_app.py` wires both to the real
`ChatOpenAI` round trips and to `/api/*` routes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.eval.corpus import ArticleRow

__all__ = ["filter_verbatim_spans", "lexical_shortlist"]

_TOKEN_PATTERN = re.compile(r"[a-zà-ÿ0-9]+")
_MIN_TOKEN_LEN = 3
# The question is scored higher than the fiche body (SPEC §70/ADR-0020's shortlist
# design): the question names what the annotator is actually looking for, the body is
# broader context that would otherwise swamp it on term count alone.
_QUESTION_WEIGHT = 3.0
_BODY_WEIGHT = 1.0
_DEFAULT_TOP_N = 10


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_PATTERN.findall(text.lower()) if len(tok) >= _MIN_TOKEN_LEN]


def _term_weights(question: str, body_text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for tok in _tokenize(question):
        weights[tok] = weights.get(tok, 0.0) + _QUESTION_WEIGHT
    for tok in _tokenize(body_text):
        weights[tok] = weights.get(tok, 0.0) + _BODY_WEIGHT
    return weights


def lexical_shortlist(
    question: str,
    body_text: str,
    articles: Sequence[ArticleRow],
    *,
    top_n: int = _DEFAULT_TOP_N,
) -> list[str]:
    """Rank `articles` by summed term-weight overlap against `question` + `body_text`
    combined, question weighted higher, and return the top `top_n` `cid`s.

    Deliberately dumb (ADR-0020: "kept dumb ... to keep that surface small") — plain
    lowercase tokenization, no stemming, no stopword list, no embeddings. An article with
    zero overlapping terms scores 0 and is excluded rather than padding the shortlist with
    noise. Ties break on `cid` for a deterministic result.
    """
    weights = _term_weights(question, body_text)
    if not weights:
        return []
    scored: list[tuple[float, str]] = []
    for row in articles:
        score = sum(weights.get(tok, 0.0) for tok in _tokenize(row["texte"]))
        if score > 0:
            scored.append((score, row["cid"]))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [cid for _, cid in scored[:top_n]]


def filter_verbatim_spans(candidates: Sequence[str], chunk_texts: Sequence[str]) -> list[str]:
    """Keep only the candidates that are an exact substring of some text in
    `chunk_texts` — SPEC §12.2's verbatim invariant, enforced structurally rather than by
    trusting the model's quoting (ADR-0020). Blank and duplicate candidates are dropped;
    order is otherwise preserved.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen and any(candidate in text for text in chunk_texts):
            seen.add(candidate)
            kept.append(candidate)
    return kept

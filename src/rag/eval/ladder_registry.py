"""The pre-registered rung table — SPEC §12.7/§12.8, ADR-0011, #37.

**The one ordering constraint in the whole build that cannot be recovered afterwards.**
Each rung and pre-ladder A/B names its **one** primary metric here, in this commit, before
a single one of them runs. Choosing a rung's metric after seeing its numbers is post-hoc
metric selection — the exact failure this table exists to prevent (ADR-0011). `compare.py`
(#36) reads the primary metric from `LADDER_TABLE` via `primary_metric_for` rather than
accepting it as a CLI argument a human could still retype after the fact.

**Rung 1 is deliberately unregistered** (`primary=None`): SPEC §12.7 calls it a "reference
floor, not a comparison" — there is nothing to adopt it against, so `primary_metric_for`
raises for it exactly like it would for a rung typo'd or never added to this table.

**Rungs 2-6 mirror SPEC §12.7's table**; `ab_article_breadcrumb` and `ab_fiche_header`
mirror §12.8's two pre-ladder A/Bs, judged under the same adoption rule and given no
seventh rung of their own (§12.8: "they cost no statistical power").

The `guard` column records the SPEC table's named guard for documentation only —
`compare.compare_runs` already checks *every* other decision metric per ADR-0011 rule 2
("no other decision metric regresses by more than 1 net item"), not just the one named
here.

This module deliberately does not import `rag.eval.compare`: `compare.py` imports this
module to resolve a run's primary metric, and `compare_runs` already rejects any primary
outside `DECISION_METRICS` (SPEC §12.7's four-metric table) on its own, so re-checking that
here would only risk a circular import for no new guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LADDER_TABLE", "RungSpec", "primary_metric_for"]


@dataclass(frozen=True)
class RungSpec:
    rung: str
    variable: str
    primary: str | None
    guard: str | None


# SPEC §12.7's table, plus §12.8's two pre-ladder A/Bs — committed as one file so both fall
# under "the same adoption rule" (#37's acceptance criteria).
LADDER_TABLE: tuple[RungSpec, ...] = (
    RungSpec(
        rung="rung1",
        variable="naive baseline — single index, dense-only, no expansion, no rerank, top-8",
        primary=None,
        guard=None,
    ),
    RungSpec(rung="rung2", variable="+ hybrid sparse leg", primary="article_recall_at_4", guard="fiche_recall_at_4"),
    RungSpec(
        rung="rung3", variable="+ <dc:source> expansion", primary="article_recall_at_4", guard="fiche_recall_at_4"
    ),
    RungSpec(rung="rung4", variable="+ reranker", primary="fiche_recall_at_4", guard="article_recall_at_4"),
    RungSpec(
        rung="rung5", variable="quota vs free-for-all", primary="zero_articles", guard="article_recall_at_4"
    ),
    RungSpec(
        rung="rung6",
        variable="embedder A/B (e5-dense + M3-sparse)",
        primary="fiche_recall_at_4",
        guard="article_recall_at_4",
    ),
    RungSpec(
        rung="ab_article_breadcrumb",
        variable="articles: raw text vs fullSectionsTitre-enriched dense / raw sparse",
        primary="article_recall_at_4",
        guard=None,
    ),
    RungSpec(
        rung="ab_fiche_header",
        variable="fiches: raw text vs title/chapitre_titre/cas_label-enriched dense / raw sparse",
        primary="fiche_recall_at_4",
        guard=None,
    ),
)

_BY_RUNG: dict[str, RungSpec] = {spec.rung: spec for spec in LADDER_TABLE}


def primary_metric_for(rung: str) -> str:
    """The pre-registered primary metric for `rung` (a `RunHeader.rung` value).

    Raises `ValueError` if `rung` is absent from `LADDER_TABLE` altogether, or is registered
    with no primary (rung1 — SPEC §12.7's "reference floor, not a comparison"). Both are the
    same refusal from `compare.py`'s side: there is no pre-registered verdict to report.
    """
    spec = _BY_RUNG.get(rung)
    if spec is None:
        raise ValueError(f"{rung!r} is not in the pre-registered ladder table (rag.eval.ladder_registry)")
    if spec.primary is None:
        raise ValueError(f"{rung!r} has no pre-registered primary metric ({spec.variable} — not a comparison)")
    return spec.primary

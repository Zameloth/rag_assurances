"""Floor-calibration candidate drafter (#95, #84, ADR-0020).

Chains the annotation helper's propose functions over a seeded random sample of fiches no
golden-set item uses, so the human step (#97) is validation only — no corpus search.

**ADR-0020's hard constraint:** gold labels come from `propose.lexical_shortlist` + the LLM
pick only. Nothing in this module's import graph reaches `rag.retrieval`, the embedder or
the reranker (asserted in `tests/eval/test_floor_calibration_draft.py`) — labels chosen by
the reranker would calibrate the reranker against itself.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag.config import Settings
from rag.eval.annotate_app import (
    _article_row_json,
    _article_shortlist,
    _draft_questions,
    _DraftError,
    _invoke,
    _propose_expected_points,
    _propose_gold_articles,
    _propose_gold_spans,
    _require_generation_settings,
)
from rag.eval.corpus import fiche_chunk_texts, fiche_ids
from rag.eval.disagree import Disagreement, run_disagreement_pass
from rag.eval.propose import filter_verbatim_spans
from rag.eval.schema import GoldenItem, dump_golden_set

__all__ = [
    "Consideration",
    "Draft",
    "DraftRun",
    "FicheDraftError",
    "draft_fiche",
    "draft_floor_calibration",
    "sample_unused_fiches",
    "write_drafts",
]

AI_ASSISTED_TAG = "ai_assisted"
# `gs-<digits>` is the id shape the golden-set validator enforces; the calibration drafts
# start well above the golden set's own range so the two never read as the same item.
FIRST_ITEM_NUMBER = 901


class FicheDraftError(Exception):
    """One fiche could not be drafted (dead LLM call, no question drafted). The run records
    it and carries on rather than losing the other fiches' spend."""


@dataclass(frozen=True)
class Consideration:
    """One shortlist candidate the drafter weighed: where it came from (`dc:source`,
    `lexical`), whether the LLM picked or rejected it, and its stated reason."""

    cid: str
    provenance: tuple[str, ...]
    verdict: str
    justification: str


@dataclass(frozen=True)
class Draft:
    """One drafted calibration candidate, for the reviewer (#97) to validate. `considered`
    is the full shortlist (lexical top-N ∪ `<dc:source>` articles) with per-candidate
    verdicts, so a sans-article claim is judgeable without searching the corpus."""

    item: GoldenItem
    considered: tuple[Consideration, ...]


def sample_unused_fiches(fiches_dir: Path, golden_items: Sequence[GoldenItem], *, n: int, seed: int) -> list[str]:
    """`n` fiche ids drawn with `random.Random(seed)` from the fiches no golden item lists
    in `gold_fiches` (capped at however many remain). The pool is sorted before drawing so
    the same seed gives the same sample whatever the filesystem's listing order."""
    used = {fiche_id for item in golden_items for fiche_id in item.gold_fiches}
    pool = sorted(fiche_ids(fiches_dir) - used)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def draft_fiche(
    fiche_id: str,
    item_id: str,
    *,
    fiches_dir: Path,
    articles: Sequence[dict[str, Any]],
    settings: Settings,
) -> Draft:
    """Chain the annotation helper's propose functions over one fiche into a candidate.

    The first drafted question is the one kept. `expected_state` defaults deterministically
    from whether the article proposal came back empty (ADR-0020), exactly as the helper does.
    """
    chunks = fiche_chunk_texts(fiche_id, fiches_dir)
    try:
        questions = _draft_questions("\n\n".join(chunks), settings)
    except _DraftError as exc:
        raise FicheDraftError(f"{fiche_id}: {exc}") from exc
    if not questions:
        raise FicheDraftError(f"{fiche_id}: the model drafted no question")
    question = questions[0]
    section_cids, lexical_cids = _article_shortlist(fiche_id, question, fiches_dir, articles)
    shortlist = section_cids | lexical_cids
    articles_by_cid = {row["cid"]: _article_row_json(row) for row in articles}
    picks = _propose_gold_articles(question, shortlist, articles_by_cid, settings) if shortlist else []
    gold_articles = tuple(cid for cid, _ in picks)
    reject_reasons = _justify_rejections(question, shortlist - set(gold_articles), articles_by_cid, settings)
    pick_reasons = dict(picks)
    considered = tuple(
        Consideration(
            cid=cid,
            provenance=(
                *(("dc:source",) if cid in section_cids else ()),
                *(("lexical",) if cid in lexical_cids else ()),
            ),
            verdict="pick" if cid in pick_reasons else "reject",
            justification=pick_reasons[cid] if cid in pick_reasons else reject_reasons.get(cid, ""),
        )
        for cid in sorted(shortlist)
    )
    spans = tuple(filter_verbatim_spans(_propose_gold_spans(question, chunks, settings), chunks))
    titles = [f"{articles_by_cid[cid]['citation_id']} — {articles_by_cid[cid]['title']}" for cid in gold_articles]
    points = tuple(_propose_expected_points(question, spans, titles, settings))
    item = GoldenItem(
        id=item_id,
        question=question,
        history=(),
        expected_state="reponse" if gold_articles else "reponse_sans_article",
        gold_fiches=(fiche_id,),
        gold_spans=spans,
        gold_articles=gold_articles,
        expected_points=points,
        tags=(AI_ASSISTED_TAG,),
    )
    return Draft(item=item, considered=considered)


_REJECT_PROMPT = (
    "Voici une question posée par un consommateur et une liste d'articles de loi qui ont été "
    "examinés puis écartés comme réponse à cette question, chacun identifié par un identifiant "
    "entre crochets.\n\n"
    "Question : {question}\n\nArticles écartés :\n{listing}\n\n"
    "Pour chaque article, réponds sur une ligne commençant exactement par l'identifiant recopié "
    "tel quel depuis son crochet (sans les crochets), suivi d'un espace, d'un tiret, puis de la "
    "raison courte pour laquelle il ne répond pas à la question. N'invente aucun identifiant."
)


def _justify_rejections(
    question: str,
    rejected_cids: set[str],
    articles_by_cid: Mapping[str, dict[str, Any]],
    settings: Settings,
) -> dict[str, str]:
    """Why each considered-but-not-picked article was left out, so a sans-article claim can
    be judged from the draft alone (#95) — `_propose_gold_articles` only reports its picks.

    Same membership guard as the pick parser: a line naming a cid outside `rejected_cids`
    is dropped, never trusted. A candidate the model skipped simply has no entry.
    """
    if not rejected_cids:
        return {}
    _require_generation_settings(settings)
    listing = "\n\n".join(
        f"[{cid}] {articles_by_cid[cid]['citation_id']} — {articles_by_cid[cid]['title']}\n"
        f"{articles_by_cid[cid]['text'][:500]}"
        for cid in sorted(rejected_cids)
    )
    content = _invoke(_REJECT_PROMPT.format(question=question, listing=listing), settings)
    reasons: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = next((cid for cid in rejected_cids if cid in line), None)
        if match is not None and match not in reasons:
            reasons[match] = line[line.index(match) + len(match) :].strip(" []:-–—\t")
    return reasons


@dataclass(frozen=True)
class DraftRun:
    """Everything one drafter run produced: the seed it is reproducible from, the drafts,
    the independent disagreement pass's findings keyed by item id (items that agree are
    absent), and the fiches that failed to draft."""

    seed: int
    drafts: list[Draft]
    disagreements: dict[str, Disagreement]
    failures: dict[str, str]


DisagreementPass = Callable[[Sequence[GoldenItem], Path, Sequence[Any], Settings], list[Disagreement]]


def draft_floor_calibration(
    *,
    fiches_dir: Path,
    articles: Sequence[dict[str, Any]],
    golden_items: Sequence[GoldenItem],
    settings: Settings,
    n: int,
    seed: int,
    disagreement_pass: DisagreementPass = run_disagreement_pass,
) -> DraftRun:
    """Draft a candidate per fiche of a seeded sample of fiches the golden set never uses,
    then run `rag.eval.disagree` over every draft as the second opinion (ADR-0010)."""
    drafts: list[Draft] = []
    failures: dict[str, str] = {}
    for fiche_id in sample_unused_fiches(fiches_dir, golden_items, n=n, seed=seed):
        item_id = f"gs-{FIRST_ITEM_NUMBER + len(drafts)}"
        try:
            drafts.append(draft_fiche(fiche_id, item_id, fiches_dir=fiches_dir, articles=articles, settings=settings))
        except FicheDraftError as exc:
            failures[fiche_id] = str(exc)
    found = disagreement_pass([d.item for d in drafts], fiches_dir, articles, settings) if drafts else []
    return DraftRun(
        seed=seed,
        drafts=drafts,
        disagreements={d.item_id: d for d in found},
        failures=failures,
    )


def write_drafts(run: DraftRun, golden_path: Path, review_path: Path) -> None:
    """Write the candidates as a golden-schema file (`history: []`, the seed in a leading
    comment — a YAML list has no room for a metadata key and still loads as the schema) and
    a review sidecar carrying what the schema has no field for: per item, the considered
    shortlist with verdicts and the disagreement-pass finding."""
    dump_golden_set([d.item for d in run.drafts], golden_path)
    body = golden_path.read_text(encoding="utf-8")
    golden_path.write_text(f"# seed: {run.seed} (rag.eval.floor_calibration_draft)\n{body}", encoding="utf-8")

    review = {
        "seed": run.seed,
        "failures": run.failures,
        "items": [
            {
                "id": d.item.id,
                "fiche": d.item.gold_fiches[0],
                "expected_state": d.item.expected_state,
                "considered": [
                    {
                        "cid": c.cid,
                        "provenance": list(c.provenance),
                        "verdict": c.verdict,
                        "justification": c.justification,
                    }
                    for c in d.considered
                ],
                "disagreement": _disagreement_entry(run.disagreements.get(d.item.id)),
            }
            for d in run.drafts
        ],
    }
    review_path.write_text(
        yaml.safe_dump(review, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100),
        encoding="utf-8",
    )


def _disagreement_entry(disagreement: Disagreement | None) -> dict[str, Any] | None:
    if disagreement is None:
        return None
    return {
        "gold_articles": sorted(disagreement.gold_articles),
        "model_picks": sorted(disagreement.model_picks),
        "only_in_gold": sorted(disagreement.only_in_gold),
        "only_in_model": sorted(disagreement.only_in_model),
        "needs_review": disagreement.needs_review,
    }

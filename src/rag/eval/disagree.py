"""Independent disagreement-detector pass over the finished golden set (ADR-0010, ADR-0020, #34).

**A separate, later pass** — never run during annotation, and sharing no code path with
`propose.py`'s lexical shortlist or `annotate_app.py`'s proposal prompt: reusing either
would make this "redundant with itself, not independent" (ADR-0020). For each
retrieval-bearing item, a model picks gold articles from *only* the union of its
`gold_fiches`' `<dc:source>` sections — the exact reading-aid pool ADR-0010 defines — blind
to the annotator's already-saved `gold_articles`. Substitutes for the inter-annotator
agreement a solo annotator can't otherwise get; it catches attention slipping on item 34 of
38, not bias (ADR-0010).

**This module never writes to `eval/golden/golden-set.yaml`.** A disagreement is reported
for human re-review, never auto-resolved — the pass has no authority to change a label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag.config import Settings
from rag.eval.corpus import ArticleRow, fiche_sections
from rag.eval.schema import GoldenItem
from rag.generation.chain import OPENROUTER_BASE_URL

__all__ = [
    "Disagreement",
    "DisagreementCallError",
    "find_outside_section_items",
    "independent_section_pick",
    "run_disagreement_pass",
    "section_shortlist",
]


@dataclass(frozen=True)
class Disagreement:
    """One item where the model's independent pick didn't exactly match the annotated
    `gold_articles`. Carries enough to tell "expected" disagreement (ADR-0010's outside-
    section freedom) apart from a genuine attention-slip candidate, without deciding which
    one it is — that call stays with the human reviewer."""

    item_id: str
    gold_articles: frozenset[str]
    model_picks: frozenset[str]
    section_cids: frozenset[str]

    @property
    def only_in_gold(self) -> frozenset[str]:
        return self.gold_articles - self.model_picks

    @property
    def only_in_model(self) -> frozenset[str]:
        return self.model_picks - self.gold_articles

    @property
    def explained_by_outside_section(self) -> frozenset[str]:
        """The subset of `only_in_gold` the model could never have picked — it wasn't in
        the section pool it was shown. Evidence the reading aid was used as a reading aid
        (ADR-0010), not an annotator error."""
        return self.only_in_gold - self.section_cids

    @property
    def needs_review(self) -> bool:
        """True when some part of the disagreement *isn't* explained by the model simply
        never having seen the article — the case worth a human's attention."""
        return bool(self.only_in_model or (self.only_in_gold - self.explained_by_outside_section))


class DisagreementCallError(Exception):
    """The independent-pick LLM round trip failed for one item. Kept distinct so a caller
    can report which item didn't complete rather than losing the whole pass to one dead
    call."""


def section_shortlist(
    item: GoldenItem, fiches_dir: Path, articles: Sequence[ArticleRow]
) -> set[str]:
    """The cid pool the disagreement pass shows the model for `item`: the union, over
    every fiche in `item.gold_fiches`, of that fiche's `<dc:source>` section articles.

    Deliberately excludes `propose.py`'s lexical shortlist — the disagreement pass only
    ever sees the same reading aid ADR-0010 describes, never anything derived from the
    item's own question the way the during-annotation proposal is."""
    cids: set[str] = set()
    for fiche_id in item.gold_fiches:
        for section in fiche_sections(fiche_id, fiches_dir, articles):
            cids.update(row["cid"] for row in section.articles)
    return cids


def find_outside_section_items(
    items: Sequence[GoldenItem], fiches_dir: Path, articles: Sequence[ArticleRow]
) -> dict[str, frozenset[str]]:
    """The structural cross-check for ADR-0010's acceptance criterion: "at least one
    item's gold article lies outside the sections its fiche cites." Returns
    `{item_id: {those cids}}` for every item where it happened — empty if the criterion
    isn't met yet.

    Computed directly from the corpus, no LLM involved: a `gold_articles` cid absent from
    `section_shortlist(item, ...)` is, by definition, outside every section the item's
    fiches cite.
    """
    result: dict[str, frozenset[str]] = {}
    for item in items:
        if not item.gold_articles:
            continue
        outside = set(item.gold_articles) - section_shortlist(item, fiches_dir, articles)
        if outside:
            result[item.id] = frozenset(outside)
    return result


_INDEPENDENT_PICK_PROMPT = (
    "Voici une question posée par un consommateur et une liste d'articles de loi, chacun "
    "identifié par un identifiant entre crochets, par exemple [LEGIARTI000006792738].\n\n"
    "Question : {question}\n\nArticles :\n{listing}\n\n"
    "Pour chaque article qui répond réellement à la question, réponds sur une ligne "
    "commençant exactement par l'identifiant recopié tel quel depuis son crochet ci-dessus "
    "(sans les crochets), suivi d'un espace, d'un tiret, puis d'une justification courte. "
    "N'invente jamais un identifiant en dehors de cette liste et ne cite aucun article de "
    'ta propre connaissance. Si aucun article ne convient, réponds "aucun".'
)


def independent_section_pick(
    question: str,
    section_cids: set[str],
    articles_by_cid: Mapping[str, ArticleRow],
    settings: Settings,
) -> set[str]:
    """One independent LLM pick, scoped to `section_cids` only.

    Same "verify membership, never trust the model's own claim" guard as
    `annotate_app._propose_gold_articles` against a hallucinated cid, but reimplemented
    here rather than imported from it — this pass must share no code path with the
    during-annotation proposal (ADR-0020) to actually be independent of it.
    """
    if not section_cids:
        return set()
    listing = "\n\n".join(
        f"[{cid}] {articles_by_cid[cid]['citation_id']} — {articles_by_cid[cid].get('ref') or ''}\n"
        f"{articles_by_cid[cid]['texte'][:500]}"
        for cid in sorted(section_cids)
        if cid in articles_by_cid
    )
    prompt = _INDEPENDENT_PICK_PROMPT.format(question=question, listing=listing)
    llm = ChatOpenAI(
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.generation_model,
        base_url=OPENROUTER_BASE_URL,
    )
    try:
        response = llm.invoke([HumanMessage(prompt)])
    except Exception as exc:  # noqa: BLE001 - surfaced to the CLI, not a stack trace
        raise DisagreementCallError(f"appel LLM échoué : {exc}") from exc
    return _parse_picks(str(response.content), section_cids)


def _parse_picks(content: str, section_cids: set[str]) -> set[str]:
    picks: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = next((cid for cid in section_cids if cid in line), None)
        if match is not None:
            picks.add(match)
    return picks


def run_disagreement_pass(
    items: Sequence[GoldenItem],
    fiches_dir: Path,
    articles: Sequence[ArticleRow],
    settings: Settings,
) -> list[Disagreement]:
    """Run the independent pick over every item with at least one gold fiche, returning
    one `Disagreement` per item whose model pick doesn't exactly match its annotated
    `gold_articles`. An item with no disagreement is simply absent from the result.

    Never touches `eval/golden/golden-set.yaml` — reporting only.
    """
    articles_by_cid = {row["cid"]: row for row in articles}
    disagreements = []
    for item in items:
        if not item.gold_fiches:
            continue
        section_cids = section_shortlist(item, fiches_dir, articles)
        model_picks = independent_section_pick(
            item.question, section_cids, articles_by_cid, settings
        )
        gold = set(item.gold_articles)
        if gold != model_picks:
            disagreements.append(
                Disagreement(
                    item_id=item.id,
                    gold_articles=frozenset(gold),
                    model_picks=frozenset(model_picks),
                    section_cids=frozenset(section_cids),
                )
            )
    return disagreements

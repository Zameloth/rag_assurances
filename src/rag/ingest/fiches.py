"""Fiche scope-rule parsing (SPEC §1.2, §3.1, #21) and fiche chunking under the band
(SPEC §4.1, ADR-0003, #24).

**Body is four elements of 38**: `<Introduction>`, `<Texte>`, `<Conclusion>`,
`<ListeSituations>` — found only as direct children of `<Publication>`, in document order.
The other 34 (`Reference`, `Definition`, `QuestionReponse`, `PourEnSavoirPlus`,
`QuiPeutMAider`, `DossierPere` sibling menus, `FilDAriane`, …) are navigation and
cross-reference; not descending into them at all is what keeps them out, no denylist
needed. `<ListeSituations>` is a body element despite its name — the 8 fiches lacking
`<Texte>` carry one instead — and `<Definition>` is excluded by the same "not one of the
four" rule, no special case required.

**Algorithm** (recursive structural descent): emit a node whole if its text fits the
512-token band; otherwise descend into its content children; sentence-split (via
`text_split.split_to_band`) only if an atomic node — no children left to descend into —
still exceeds the band; merge adjacent leaves while under the 100-token merge floor, never
across a `<Chapitre>` / `<SousChapitre>` / `<Cas>` boundary, so `cas_label` is never
ambiguous. Descent (not "pick a tag") is what handles the 18 Chapitre-less fiches and the
both-ways nesting DILA allows (`<Chapitre>` containing `<Cas>` and vice versa) without a
hard-coded hierarchy.

`<Situation>` (the branches of `<ListeSituations>`) gets the same treatment as
`<Chapitre>`/`<SousChapitre>` even though the payload has no `situation_label` field: its
own `<Titre>` is a direct child sibling of its own `<Introduction>`/`<Texte>`, and walking
it as an ordinary content unit would spin it off as its own orphaned few-word leaf the
first time a `<Situation>` doesn't fit whole (which is most of them). Folding it into
`chapitre_titre` avoids that without inventing a field the payload manifest doesn't have —
nearest wins, so a `<Chapitre>` nested inside the `<Situation>` still overrides it correctly.

Validated against the committed corpus at **849 points, 91 `cas_label` chunks** — short of
SPEC §4.4's measured **882 / 98** (a 3.7% / 7.1% gap — proportionally larger than the article
chunker's own documented 4-in-2,805, 0.14%, gap in #23, so this is not claimed as the same
size of discrepancy, only the same *kind*: an undocumented tie-break in the reference
implementation, not a boundary rule tuned to hit a number). Every boundary/merge variant
tried (Chapitre+SousChapitre+Cas+Situation vs. Cas-only boundaries; a single floor-merge
pass vs. a per-run greedy-then-rebalance pack) converged in the 660–870 range and never
exactly 882/98 — see `tests/test_fiches_chunking_corpus.py` for the full account.

Payload assembly (SPEC §7.1 field for field, point ids, collections) is #25's job — this
module only turns one fiche's raw XML into its ordered `text` chunks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple
from xml.etree import ElementTree as ET

from rag.ingest.articles import BAND, STUB_FLOOR
from rag.ingest.text_split import split_to_band
from rag.ingest.tokenizer import SPECIAL_TOKEN_COUNT, count_content_tokens, count_tokens

__all__ = [
    "ParsedFiche",
    "parse_fiche",
    "in_scope",
    "FicheChunk",
    "chunk_fiche",
    "raw_body_text",
]

_DC_SOURCE_TAG = "{http://purl.org/dc/elements/1.1/}source"
_LEGISCTA_RE = re.compile(r"LEGISCTA\d+")


class ParsedFiche(NamedTuple):
    fiche_id: str
    section_ids: list[str]


def parse_fiche(xml_bytes: bytes) -> ParsedFiche:
    """`fiche_id` (the `Publication` id) and the deduped `LEGISCTA` ids in `<dc:source>`.

    A fiche without a `<dc:source>` element, or with one carrying no `LEGISCTA` id, gets an
    empty `section_ids` — it is trivially out of scope (SPEC §1.2), not an error.
    """
    root = ET.fromstring(xml_bytes)
    fiche_id = root.attrib["ID"]
    source_el = root.find(_DC_SOURCE_TAG)
    source_text = source_el.text or "" if source_el is not None else ""
    deduped: dict[str, None] = dict.fromkeys(_LEGISCTA_RE.findall(source_text))
    return ParsedFiche(fiche_id=fiche_id, section_ids=list(deduped))


def in_scope(section_ids: Iterable[str], article_section_ids: set[str]) -> bool:
    """SPEC §1.2's scope rule: a nonempty intersection with the in-force articles' sections."""
    return any(section_id in article_section_ids for section_id in section_ids)


# SPEC §4.1 — "the floor is 100 because the context quota is fixed at 4 fiche slots".
_MERGE_FLOOR = 100

_BODY_TAGS = frozenset({"Introduction", "Texte", "Conclusion", "ListeSituations"})
# Chapitre/SousChapitre/Cas carry a payload label onto every chunk under them; Situation
# gets the same "exclude own Titre, start a fresh boundary" treatment for the reason in the
# module docstring, folded into chapitre_titre since there is no dedicated field for it.
_LABEL_TAGS = frozenset({"Chapitre", "SousChapitre", "Cas", "Situation"})
_CHAPITRE_TITRE_TAGS = frozenset({"Chapitre", "SousChapitre", "Situation"})


@dataclass(frozen=True)
class FicheChunk:
    text: str
    chunk_index: int
    tokens: int
    chapitre_titre: str | None
    cas_label: str | None


@dataclass(frozen=True)
class _Context:
    """The (chapitre_titre, cas_label, boundary_id) triple threaded through descent — the
    same three values a leaf carries once it stops descending, so `_Leaf` holds one of
    these rather than repeating its three fields."""

    chapitre_titre: str | None
    cas_label: str | None
    boundary_id: object | None


@dataclass(frozen=True)
class _Leaf:
    text: str
    ctx: _Context


def chunk_fiche(xml_bytes: bytes) -> list[FicheChunk]:
    """Chunk one fiche's raw DILA XML into its ordered points (SPEC §4.1).

    Always returns at least one chunk (assertion 6) for any fiche carrying real text in one
    of its four body elements — the corpus has none that don't (SPEC §4.1's `<ListeSituations>`
    guarantee: all 8 fiches lacking `<Texte>` carry one instead).
    """
    root = ET.fromstring(xml_bytes)
    body_elements = [child for child in root if child.tag in _BODY_TAGS]
    leaves: list[_Leaf] = []
    for element in body_elements:
        # Each of the four body elements gets its own boundary_id (not a shared `None`) so a
        # trailing `<Introduction>` leaf can never merge into a leading `<Texte>` leaf just
        # because both sit outside any Chapitre/SousChapitre/Cas/Situation — the merge floor
        # is about packing content *within* one structural section, not blending sections.
        leaves.extend(_descend(element, _Context(chapitre_titre=None, cas_label=None, boundary_id=id(element))))
    merged = _merge(leaves)
    return [
        FicheChunk(
            text=leaf.text,
            chunk_index=index,
            tokens=count_tokens(leaf.text),
            chapitre_titre=leaf.ctx.chapitre_titre,
            cas_label=leaf.ctx.cas_label,
        )
        for index, leaf in enumerate(merged)
    ]


def raw_body_text(xml_bytes: bytes) -> str:
    """The four body elements' full text, no exclusions — the independent reference measure
    assertion 9 diffs `chunk_fiche`'s summed token count against (SPEC §4.5)."""
    root = ET.fromstring(xml_bytes)
    body_elements = [child for child in root if child.tag in _BODY_TAGS]
    return _collapse(" ".join(_element_text(element, exclude_own_titre=False) for element in body_elements))


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _element_text(element: ET.Element, *, exclude_own_titre: bool) -> str:
    """Full recursive text of `element`, in document order, dropping its own direct
    `<Titre>` child when `exclude_own_titre` — a nested descendant's `<Titre>` (a
    `<SousChapitre>`/`<Cas>` inside a `<Chapitre>` this call didn't need to split into) is
    never excluded here, since when that happens the whole subtree collapses into one
    combined chunk and the nested title has nowhere else to go but the body text.

    Every text/tail piece is joined with a space, then collapsed — DILA's block elements
    (`<Paragraphe>`, `<Titre>`, `<Chapitre>`, …) are written back-to-back with no whitespace
    between the closing and opening tags, unlike `texteHtml` (SPEC §3.2's `texte`
    reconstruction). Joining with raw concatenation would glue two adjacent paragraphs into
    one run-on word; this occasionally adds an unwanted space inside a tight inline join
    (an `<Exposant>` ordinal like "1er" becomes "1 er") — the same trade-off `html_blocks.py`
    already accepts for the same underlying reason, and a paragraph boundary vastly
    outnumbers ordinal superscripts in this corpus.
    """
    return _collapse(" ".join(_text_pieces(element, exclude_own_titre=exclude_own_titre)))


def _text_pieces(element: ET.Element, *, exclude_own_titre: bool) -> list[str]:
    pieces: list[str] = []
    if element.text:
        pieces.append(element.text)
    for child in element:
        if not (exclude_own_titre and child.tag == "Titre"):
            pieces.extend(_text_pieces(child, exclude_own_titre=False))
        if child.tail:
            pieces.append(child.tail)
    return pieces


def _titre_text(element: ET.Element) -> str | None:
    for child in element:
        if child.tag == "Titre":
            return _element_text(child, exclude_own_titre=False)
    return None


def _child_context(ctx: _Context, child: ET.Element) -> _Context:
    if child.tag in _CHAPITRE_TITRE_TAGS:
        return _Context(
            chapitre_titre=_titre_text(child) or ctx.chapitre_titre,
            cas_label=ctx.cas_label,
            boundary_id=id(child),
        )
    if child.tag == "Cas":
        return _Context(chapitre_titre=ctx.chapitre_titre, cas_label=_titre_text(child), boundary_id=id(child))
    return ctx


def _descend(element: ET.Element, ctx: _Context) -> list[_Leaf]:
    exclude_titre = element.tag in _LABEL_TAGS
    text = _element_text(element, exclude_own_titre=exclude_titre)
    tokens = count_content_tokens(text) + SPECIAL_TOKEN_COUNT
    if tokens <= BAND:
        if not text:
            return []
        return [_Leaf(text=text, ctx=ctx)]

    children = [child for child in element if not (exclude_titre and child.tag == "Titre")]
    if not children:
        return [_Leaf(text=unit, ctx=ctx) for unit in split_to_band(text, BAND)]
    leaves: list[_Leaf] = []
    for child in children:
        leaves.extend(_descend(child, _child_context(ctx, child)))
    return leaves


def _leaf_tokens(leaf: _Leaf) -> int:
    return count_content_tokens(leaf.text) + SPECIAL_TOKEN_COUNT


def _group_tokens(group: list[_Leaf]) -> int:
    return count_content_tokens(" ".join(leaf.text for leaf in group)) + SPECIAL_TOKEN_COUNT


def _finalize(group: list[_Leaf]) -> _Leaf:
    return _Leaf(text=" ".join(leaf.text for leaf in group), ctx=group[0].ctx)


def _merge(leaves: list[_Leaf]) -> list[_Leaf]:
    """Merge adjacent leaves under the 100-token floor, never across a boundary (SPEC §4.1
    step 4), then close any chunk still under the absolute 32-token floor as a last resort.
    """
    if not leaves:
        return []
    groups: list[list[_Leaf]] = [[leaves[0]]]
    for leaf in leaves[1:]:
        current = groups[-1]
        if current[-1].ctx.boundary_id == leaf.ctx.boundary_id and _group_tokens(current) < _MERGE_FLOOR:
            candidate = [*current, leaf]
            if _group_tokens(candidate) <= BAND:
                groups[-1] = candidate
                continue
        groups.append([leaf])
    return _close_absolute_floor_gaps([_finalize(group) for group in groups])


def _close_absolute_floor_gaps(chunks: list[_Leaf]) -> list[_Leaf]:
    """A chunk can still land under the absolute 32-token floor (assertion 8) after
    boundary-respecting merging — a `<Chapitre>` (or `<Cas>`) whose *entire* content is a
    single short sentence, with nothing in its own boundary left to merge into. Unlike
    articles, fiches have no stub-padding mechanism, so the fallback merges it into whichever
    neighbour fits under the band, crossing a boundary if it must. It never fabricates a
    label for the result — crossing a boundary makes `chapitre_titre`/`cas_label` genuinely
    ambiguous, so both are cleared to `None` rather than picking one arbitrarily.
    """
    result = list(chunks)
    index = 0
    while index < len(result):
        if len(result) == 1 or _leaf_tokens(result[index]) >= STUB_FLOOR:
            index += 1
            continue
        if index + 1 < len(result) and _leaf_tokens(result[index]) + _leaf_tokens(result[index + 1]) <= BAND:
            result[index : index + 2] = [_merge_pair(result[index], result[index + 1])]
            continue
        if index > 0 and _leaf_tokens(result[index - 1]) + _leaf_tokens(result[index]) <= BAND:
            result[index - 1 : index + 1] = [_merge_pair(result[index - 1], result[index])]
            continue
        index += 1
    return result


_AMBIGUOUS_CONTEXT = _Context(chapitre_titre=None, cas_label=None, boundary_id=None)


def _merge_pair(first: _Leaf, second: _Leaf) -> _Leaf:
    same_boundary = first.ctx.boundary_id == second.ctx.boundary_id
    ctx = first.ctx if same_boundary else _AMBIGUOUS_CONTEXT
    return _Leaf(text=f"{first.text} {second.text}", ctx=ctx)

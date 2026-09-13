"""The golden-set annotation helper (SPEC §12.1-§12.4, ADR-0010, #33, #44).

A local FastAPI app, read-only against `data/corpus/` — it only ever writes
`eval/golden/golden-set.yaml` and its id counter file. Two authoring surfaces:

- `/fiche/{fiche_id}` (#33, #70): one screen per fiche, LLM-assisted — the fiche's own
  chunk text (for highlighting a `gold_span` verbatim by construction), its `<dc:source>`
  sections' in-force articles as a reading list, and full-corpus article search — both
  rendered through the **same** client-side row template, so picking an article the fiche
  never cited is never the harder path (SPEC §12.3).
- `/manual` (#44): the 22 behavioural items and the 10 multi-turn ones — handwritten, no
  LLM, no fiche anchor. `gold_fiches`/`gold_articles` are picked by search rather than read
  off one fiche's own text, since regulated-act refusals still need populated gold contexts
  (SPEC §12.4) even though nothing here anchors on a single fiche.

Both post to the same `/save` — `GET /save` does not exist; `POST /save` **re-validates the
whole candidate file** against the committed corpus before writing anything, using the exact
function (`validate_golden_set_against_corpus`) the CLI validator runs — the helper and the
validator can never disagree about what "valid" means because they are the same code.

Run it with:

    uv run uvicorn rag.eval.annotate_app:app --reload
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from rag.config import Settings, load_settings
from rag.eval.corpus import (
    ArticleRow,
    fiche_chunk_texts,
    fiche_sections,
    fiche_summaries,
    fiche_title,
    load_articles,
)
from rag.eval.corpus import fiche_ids as corpus_fiche_ids
from rag.eval.ids import allocate_id
from rag.eval.propose import filter_verbatim_spans, lexical_shortlist
from rag.eval.schema import (
    EXPECTED_STATES,
    MULTI_TURN_TAG,
    GoldenItem,
    dump_golden_set,
    load_golden_set,
)
from rag.eval.validate import GoldenSetValidationError, validate_golden_set_against_corpus
from rag.generation.chain import OPENROUTER_BASE_URL

__all__ = ["create_app"]

REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# SPEC §12.1's composition table, split across the two tickets that annotate it — a target
# the validator deliberately never checks (it's per-item, not corpus-wide), so the helper
# surfaces it instead as a live progress readout, not a gate. `_TARGET_TOTAL` is the full
# 60-item golden set (SPEC §12.1's "60 hand-annotated items"), not just #34's slice —
# `"total": len(items)` in `/api/progress` below counts every item in the file regardless
# of which authoring surface wrote it, so its target has to match.
_TARGET_TOTAL = 60
# The 38 retrieval-bearing items (#34):
_TARGET_REPONSE = 30
_TARGET_CITATION_FORM = 4
_TARGET_SANS_ARTICLE = 8
# The 22 behavioural items (#44) — the composition table's third/fourth rows:
_TARGET_REFUS_REGULATED = 12
_TARGET_REFUS_HORS_CORPUS = 10
# Cross-cutting across all 60 (#44) — "50 single-turn / 10 multi-turn":
_TARGET_MULTI_TURN = 10
_REFUS_REGULATED_STATES = frozenset({"refus:recommandation_produit", "refus:conseil_action"})
_CITATION_FORM_TAG = "citation_form"
_AI_ASSISTED_TAG = "ai_assisted"

# SPEC §12.4 — `reponse_sans_article` and `refus:hors_corpus` both leave `gold_articles`
# empty, and they mean opposite things ("no article clears the floor" vs. "nothing exists
# to retrieve"). The closed-vocabulary <select> is where an annotator could conflate them,
# so each option spells out which one it means instead of leaving two bare identifiers to
# tell apart from memory.
_EXPECTED_STATE_LABELS: dict[str, str] = {
    "reponse": "reponse — le corpus répond, avec au moins un gold_article",
    "reponse_sans_article": (
        "reponse_sans_article — la réponse existe mais aucun article ne suffit "
        "(gold_articles vide par construction ; gold_fiches renseigné)"
    ),
    "refus:recommandation_produit": (
        "refus:recommandation_produit — refus d'acte régulé (recommander un produit) ; "
        "gold_fiches et gold_articles restent renseignés, la retrieval réussit"
    ),
    "refus:conseil_action": (
        "refus:conseil_action — refus d'acte régulé (conseiller une action) ; "
        "gold_fiches et gold_articles restent renseignés, la retrieval réussit"
    ),
    "refus:hors_corpus": "refus:hors_corpus — rien n'existe dans le corpus (gold_fiches et gold_articles vides)",
}


def _expected_state_options() -> list[dict[str, Any]]:
    # `regulated_refus`/`no_points` travel with the option instead of being re-derived
    # (or re-hardcoded) client-side — `manual.html`'s state-change hints read them straight
    # off the selected `<option>`'s `dataset`, so the regulated-refus pair and the
    # hors_corpus/no-`expected_points` rule each have exactly one source of truth.
    return [
        {
            "value": state,
            "label": _EXPECTED_STATE_LABELS[state],
            "regulated_refus": state in _REFUS_REGULATED_STATES,
            "no_points": state == "refus:hors_corpus",
        }
        for state in sorted(EXPECTED_STATES)
    ]


# ADR-0020's shortlist size — large enough that a relevant off-`<dc:source>` article
# usually surfaces, small enough that the LLM prompt stays a handful of articles, not the
# full 2,377-article corpus.
_SHORTLIST_TOP_N = 10


class HistoryTurnIn(BaseModel):
    role: str
    content: str


class DraftQuestionsRequest(BaseModel):
    fiche_id: str


class FicheQuestionRequest(BaseModel):
    """Shared by both fiche-scoped propose endpoints — gold_articles and gold_spans each
    need only a fiche id and the annotator's current question text."""

    fiche_id: str
    question: str


class ProposeExpectedPointsRequest(BaseModel):
    question: str
    gold_spans: list[str] = []
    gold_articles: list[str] = []


class SaveRequest(BaseModel):
    question: str
    expected_state: str
    gold_fiches: list[str] = []
    gold_articles: list[str] = []
    gold_spans: list[str] = []
    expected_points: list[str] = []
    tags: list[str] = []
    history: list[HistoryTurnIn] = []
    ai_assisted: bool = False


def create_app(
    *,
    fiches_dir: Path = REPO_ROOT / "data" / "corpus" / "fiches",
    articles_path: Path = REPO_ROOT / "data" / "corpus" / "articles.jsonl",
    golden_set_path: Path = REPO_ROOT / "eval" / "golden" / "golden-set.yaml",
    id_counter_path: Path = REPO_ROOT / "eval" / "golden" / ".next_id",
) -> FastAPI:
    """Build the app against a given corpus/golden-set location.

    Defaulting the four paths to the real repo layout is what makes
    `uvicorn rag.eval.annotate_app:app` work with zero configuration; tests pass tmp_path
    fixtures instead so a test run never touches the real committed corpus or golden set.
    """
    app = FastAPI(title="Golden-set annotation helper")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # Articles never change during a session — read once rather than on every request.
    articles: Sequence[ArticleRow] = load_articles(articles_path)
    all_articles_by_cid = {row["cid"]: _article_row_json(row) for row in articles}
    # Loaded once like `articles` — a key rotated mid-session needs a restart, same as the
    # corpus does; this is a dev tool, not a long-lived server.
    settings = load_settings()

    @app.get("/api/progress")
    def api_progress() -> JSONResponse:
        items = load_golden_set(golden_set_path)
        reponse = [item for item in items if item.expected_state == "reponse"]
        return JSONResponse(
            {
                "total": len(items),
                "target_total": _TARGET_TOTAL,
                "reponse": len(reponse),
                "target_reponse": _TARGET_REPONSE,
                "citation_form": sum(1 for item in reponse if _CITATION_FORM_TAG in item.tags),
                "target_citation_form": _TARGET_CITATION_FORM,
                "sans_article": sum(1 for item in items if item.expected_state == "reponse_sans_article"),
                "target_sans_article": _TARGET_SANS_ARTICLE,
                # #44's half of SPEC §12.1's composition table — the 22 behavioural items.
                "refus_regulated": sum(1 for item in items if item.expected_state in _REFUS_REGULATED_STATES),
                "target_refus_regulated": _TARGET_REFUS_REGULATED,
                "refus_hors_corpus": sum(1 for item in items if item.expected_state == "refus:hors_corpus"),
                "target_refus_hors_corpus": _TARGET_REFUS_HORS_CORPUS,
                "multi_turn": sum(1 for item in items if MULTI_TURN_TAG in item.tags),
                "target_multi_turn": _TARGET_MULTI_TURN,
            }
        )

    @app.post("/api/draft-questions")
    def api_draft_questions(payload: DraftQuestionsRequest) -> JSONResponse:
        if payload.fiche_id not in corpus_fiche_ids(fiches_dir):
            raise HTTPException(status_code=404, detail=f"unknown fiche {payload.fiche_id}")
        body_text = "\n\n".join(fiche_chunk_texts(payload.fiche_id, fiches_dir))
        try:
            questions = _draft_questions(body_text, settings)
        except _DraftError as exc:
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=502)
        return JSONResponse({"ok": True, "questions": questions})

    @app.post("/api/propose-gold-articles")
    def api_propose_gold_articles(payload: FicheQuestionRequest) -> JSONResponse:
        if payload.fiche_id not in corpus_fiche_ids(fiches_dir):
            raise HTTPException(status_code=404, detail=f"unknown fiche {payload.fiche_id}")
        sections = fiche_sections(payload.fiche_id, fiches_dir, articles)
        section_cids = {row["cid"] for section in sections for row in section.articles}
        body_text = "\n\n".join(fiche_chunk_texts(payload.fiche_id, fiches_dir))
        lexical_cids = set(lexical_shortlist(payload.question, body_text, articles, top_n=_SHORTLIST_TOP_N))
        shortlist_cids = section_cids | lexical_cids
        if not shortlist_cids:
            return JSONResponse({"ok": True, "candidates": []})
        try:
            picks = _propose_gold_articles(payload.question, shortlist_cids, all_articles_by_cid, settings)
        except _DraftError as exc:
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=502)
        candidates = [
            {
                "cid": cid,
                "justification": justification,
                "provenance": [
                    *(["dc:source"] if cid in section_cids else []),
                    *(["lexical"] if cid in lexical_cids else []),
                ],
            }
            for cid, justification in picks
        ]
        return JSONResponse({"ok": True, "candidates": candidates})

    @app.post("/api/propose-gold-spans")
    def api_propose_gold_spans(payload: FicheQuestionRequest) -> JSONResponse:
        if payload.fiche_id not in corpus_fiche_ids(fiches_dir):
            raise HTTPException(status_code=404, detail=f"unknown fiche {payload.fiche_id}")
        chunks = fiche_chunk_texts(payload.fiche_id, fiches_dir)
        try:
            raw_candidates = _propose_gold_spans(payload.question, chunks, settings)
        except _DraftError as exc:
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=502)
        # Structural enforcement of SPEC §12.2 (ADR-0020) — a non-verbatim candidate is
        # dropped here, never shown to the annotator as a pickable span.
        spans = filter_verbatim_spans(raw_candidates, chunks)
        return JSONResponse({"ok": True, "spans": spans})

    @app.post("/api/propose-expected-points")
    def api_propose_expected_points(payload: ProposeExpectedPointsRequest) -> JSONResponse:
        article_titles = [
            f"{all_articles_by_cid[cid]['citation_id']} — {all_articles_by_cid[cid]['title']}"
            for cid in payload.gold_articles
            if cid in all_articles_by_cid
        ]
        try:
            points = _propose_expected_points(payload.question, payload.gold_spans, article_titles, settings)
        except _DraftError as exc:
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=502)
        return JSONResponse({"ok": True, "points": points})

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        summaries = fiche_summaries(fiches_dir)
        fiches_json = _safe_json([{"id": s.fiche_id, "title": s.title} for s in summaries])
        return templates.TemplateResponse(request, "index.html", {"fiches_json": fiches_json})

    @app.get("/api/articles")
    def api_articles() -> JSONResponse:
        # The full corpus (~8 MB) is one cacheable payload the browser fetches once, not
        # re-embedded in every `/fiche/{id}` page — full-text search still runs over it
        # client-side (SPEC §12.3's "just as easy to pick from outside the section").
        return JSONResponse(all_articles_by_cid, headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/fiches")
    def api_fiches() -> JSONResponse:
        # #44's no-fiche-anchor page needs a fiche *picker*, not a single anchor — same
        # (id, title) pairs `index.html` already embeds, exposed as JSON so `/manual`
        # can search them client-side instead of re-picking a page to start from.
        summaries = fiche_summaries(fiches_dir)
        return JSONResponse([{"id": s.fiche_id, "title": s.title} for s in summaries])

    @app.get("/manual", response_class=HTMLResponse)
    def manual(request: Request) -> HTMLResponse:
        # #44 — the 22 behavioural items and the 10 multi-turn ones are handwritten, not
        # fiche-anchored (SPEC §12.4): no LLM draft/propose buttons here, and gold_fiches /
        # gold_articles are picked by search rather than read off one fiche's own text.
        return templates.TemplateResponse(
            request,
            "manual.html",
            {"expected_states": _expected_state_options()},
        )

    @app.get("/fiche/{fiche_id}", response_class=HTMLResponse)
    def fiche_detail(request: Request, fiche_id: str) -> HTMLResponse:
        if fiche_id not in corpus_fiche_ids(fiches_dir):
            raise HTTPException(status_code=404, detail=f"unknown fiche {fiche_id}")

        chunks = fiche_chunk_texts(fiche_id, fiches_dir)
        sections = fiche_sections(fiche_id, fiches_dir, articles)
        sections_json = [
            {"section_id": s.section_id, "title": s.title, "cids": [row["cid"] for row in s.articles]}
            for s in sections
        ]

        return templates.TemplateResponse(
            request,
            "fiche_detail.html",
            {
                "fiche_id": fiche_id,
                "title": fiche_title((fiches_dir / f"{fiche_id}.xml").read_bytes()),
                "chunks": list(enumerate(chunks)),
                "expected_states": _expected_state_options(),
                "sections_json": _safe_json(sections_json),
            },
        )

    @app.post("/save")
    def save(payload: SaveRequest) -> JSONResponse:
        existing = load_golden_set(golden_set_path)
        new_id = allocate_id(id_counter_path, existing_ids=[item.id for item in existing])
        tags = list(payload.tags)
        if payload.ai_assisted and _AI_ASSISTED_TAG not in tags:
            tags.append(_AI_ASSISTED_TAG)
        # #44 — multi_turn is a cross-cutting *tag*, not a fifth state, derived from
        # `history` in both directions rather than trusted from the client: an annotator
        # can't forget it on a scripted follow-up, and a stray tag from a copy-pasted item
        # can't survive turning history back to `[]`.
        if payload.history and MULTI_TURN_TAG not in tags:
            tags.append(MULTI_TURN_TAG)
        elif not payload.history and MULTI_TURN_TAG in tags:
            tags.remove(MULTI_TURN_TAG)
        item = GoldenItem(
            id=new_id,
            question=payload.question,
            history=tuple({"role": turn.role, "content": turn.content} for turn in payload.history),
            expected_state=payload.expected_state,
            gold_fiches=tuple(payload.gold_fiches),
            gold_spans=tuple(payload.gold_spans),
            gold_articles=tuple(payload.gold_articles),
            expected_points=tuple(payload.expected_points),
            tags=tuple(tags),
        )
        candidate = [*existing, item]
        try:
            validate_golden_set_against_corpus(candidate, fiches_dir=fiches_dir, articles_path=articles_path)
        except GoldenSetValidationError as exc:
            # The id is burned even on a failed save (SPEC §12.1 "never reused" outranks
            # "no gaps") — nothing is written to golden_set_path.
            return JSONResponse({"ok": False, "id": new_id, "detail": str(exc)}, status_code=422)

        dump_golden_set(candidate, golden_set_path)
        return JSONResponse({"ok": True, "id": new_id})

    return app


class _DraftError(Exception):
    """An LLM round trip in the annotation helper failed — missing credentials or a dead
    call. Kept distinct from `HTTPException` so each endpoint can turn it into a 502 with
    the real reason, instead of a stack trace, in a tool the annotator watches live."""


def _require_generation_settings(settings: Settings) -> None:
    if not settings.openrouter_api_key or not settings.generation_model:
        raise _DraftError("OPENROUTER_API_KEY / GENERATION_MODEL manquant dans l'environnement (.env)")


def _invoke(prompt: str, settings: Settings) -> str:
    """One-shot `ChatOpenAI` call shared by every propose/draft helper below — same
    client construction, same failure-to-`_DraftError` mapping, so the four LLM round
    trips in this module can't drift apart on error handling."""
    llm = ChatOpenAI(
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.generation_model,
        base_url=OPENROUTER_BASE_URL,
    )
    try:
        response = llm.invoke([HumanMessage(prompt)])
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a 502, not a 500 traceback
        raise _DraftError(f"appel LLM échoué : {exc}") from exc
    return str(response.content)


def _split_lines(content: str, *, strip_chars: str = " -•\t") -> list[str]:
    """Every propose/draft helper turns one free-text LLM response into one candidate per
    line the same way — factored out so the four round trips can't drift on how a blank or
    bullet-prefixed line is handled. Filtering happens *after* stripping, so a line that is
    only a bullet character (e.g. a lone "-") drops out rather than surviving as ""."""
    return [line for line in (raw.strip(strip_chars) for raw in content.splitlines()) if line]


_DRAFT_PROMPT_PREFIX = (
    "Voici le texte d'une fiche service-public.fr sur l'assurance. Génère 2 à 3 questions "
    "qu'un consommateur (pas un juriste) poserait spontanément à un assistant, en te basant sur ce "
    "contenu. Utilise le vocabulaire d'un particulier, pas le vocabulaire juridique du texte. "
    "Réponds avec une question par ligne, sans numérotation ni tiret.\n\n"
)


def _draft_questions(body_text: str, settings: Settings) -> list[str]:
    """Ask the generation model for 2-3 consumer questions from a fiche's body text —
    the same round trip the "Copier le prompt pour le LLM" button hands to an external
    chat, done in-process to save the copy/paste (SPEC §12.4's provenance rule is
    unaffected: this drafts candidates only, the human still rewrites one by hand)."""
    _require_generation_settings(settings)
    content = _invoke(_DRAFT_PROMPT_PREFIX + body_text, settings)
    return _split_lines(content)


_PROPOSE_ARTICLES_PROMPT = (
    "Voici une question posée par un consommateur et une liste d'articles de loi candidats, "
    "chacun identifié par un identifiant entre crochets, par exemple [LEGIARTI000006792738].\n\n"
    "Question : {question}\n\nArticles candidats :\n{listing}\n\n"
    "Pour chaque article qui répond réellement à la question, réponds sur une ligne commençant "
    "exactement par l'identifiant recopié tel quel depuis son crochet ci-dessus (sans les "
    "crochets), suivi d'un espace, d'un tiret, puis d'une justification courte — par exemple "
    "`LEGIARTI000006792738 - définit l'assurance de protection juridique`. N'invente jamais un "
    "identifiant en dehors de cette liste et ne cite aucun article de ta propre connaissance. Si "
    "aucun article ne convient, réponds \"aucun\"."
)


def _propose_gold_articles(
    question: str,
    shortlist_cids: set[str],
    articles_by_cid: Mapping[str, dict[str, Any]],
    settings: Settings,
) -> list[tuple[str, str]]:
    """Ask the generation model to pick, from `shortlist_cids` only, which articles answer
    `question`, each with a short justification.

    The structural guard against a hallucinated `citation_id`/`cid` (ADR-0020) lives here:
    any line naming a cid outside `shortlist_cids` is dropped before it ever reaches the
    caller — the same "verify, don't trust the model's own claim" shape as
    `filter_verbatim_spans` for gold_spans.
    """
    _require_generation_settings(settings)
    listing = "\n\n".join(
        f"[{cid}] {articles_by_cid[cid]['citation_id']} — {articles_by_cid[cid]['title']}\n"
        f"{articles_by_cid[cid]['text'][:500]}"
        for cid in sorted(shortlist_cids)
    )
    prompt = _PROPOSE_ARTICLES_PROMPT.format(question=question, listing=listing)
    content = _invoke(prompt, settings)
    picks: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Search for a shortlist cid anywhere in the line rather than assuming the model
        # kept to the exact requested format (it sometimes echoes the prompt's own "[cid]"
        # bracket notation, or adds a stray label first) — the structural guard against a
        # hallucinated cid (ADR-0020) is the membership check, not the parsing strictness.
        match = next((cid for cid in shortlist_cids if cid in line), None)
        # A repeated cid across two lines (the model re-justifying itself) would otherwise
        # surface as two identical candidates in the UI — same dedup shape as
        # `filter_verbatim_spans` uses for gold_spans.
        if match is None or match in seen:
            continue
        seen.add(match)
        justification = line[line.index(match) + len(match) :].strip(" []:-–—\t")
        picks.append((match, justification))
    return picks


_PROPOSE_SPANS_PROMPT = (
    "Voici le texte d'une fiche service-public.fr et une question posée par un consommateur.\n\n"
    "Question : {question}\n\nTexte :\n{body_text}\n\n"
    "Recopie 2 à 4 extraits du texte ci-dessus, mot pour mot et sans les modifier ni les "
    "raccourcir, qui permettraient de répondre à la question. Un extrait par ligne, sans "
    "guillemets, sans numérotation, sans '...'."
)


def _propose_gold_spans(question: str, chunks: Sequence[str], settings: Settings) -> list[str]:
    """Ask the generation model for candidate verbatim quotes — unfiltered. The caller
    (`api_propose_gold_spans`) runs these through `filter_verbatim_spans` before they ever
    reach the UI; this function does not itself guarantee verbatim-ness."""
    _require_generation_settings(settings)
    prompt = _PROPOSE_SPANS_PROMPT.format(question=question, body_text="\n\n".join(chunks))
    content = _invoke(prompt, settings)
    return _split_lines(content, strip_chars=' -•\t"')


_PROPOSE_POINTS_PROMPT = (
    "Voici une question posée par un consommateur et le contexte retenu pour y répondre.\n\n"
    "Question : {question}\n\n{context}\n\n"
    "Rédige 1 à 3 affirmations courtes et vérifiables qu'une bonne réponse devrait "
    "obligatoirement énoncer. Une affirmation par ligne, sans numérotation."
)
_MAX_PROPOSED_POINTS = 3


def _propose_expected_points(
    question: str,
    spans: Sequence[str],
    article_titles: Sequence[str],
    settings: Settings,
) -> list[str]:
    _require_generation_settings(settings)
    context_parts = []
    if spans:
        context_parts.append("Extraits retenus :\n" + "\n".join(f"- {span}" for span in spans))
    if article_titles:
        context_parts.append("Articles retenus :\n" + "\n".join(f"- {title}" for title in article_titles))
    context = "\n\n".join(context_parts) if context_parts else "(aucun contexte sélectionné pour l'instant)"
    prompt = _PROPOSE_POINTS_PROMPT.format(question=question, context=context)
    content = _invoke(prompt, settings)
    return _split_lines(content)[:_MAX_PROPOSED_POINTS]


def _article_row_json(row: ArticleRow) -> dict[str, Any]:
    return {
        "cid": row["cid"],
        "citation_id": row["citation_id"],
        "title": row.get("ref") or row["citation_id"],
        "text": row["texte"],
        "section_id": row.get("sectionParentId") or "",
    }


def _safe_json(value: Any) -> str:
    """`json.dumps`, with `</script` neutralised so embedding the result inside a
    `<script>` tag can never prematurely close it — French insurance article text is free
    text, not something this module controls the contents of."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


# The module-level instance `uvicorn rag.eval.annotate_app:app` serves — the real corpus
# and golden set, at their default repo-relative locations.
app = create_app()

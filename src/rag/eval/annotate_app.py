"""The golden-set annotation helper (SPEC §12.1-§12.4, ADR-0010, #33).

A local FastAPI app, read-only against `data/corpus/` — it only ever writes
`eval/golden/golden-set.yaml` and its id counter file. One screen per fiche: the fiche's
own chunk text (for highlighting a `gold_span` verbatim by construction), its `<dc:source>`
sections' in-force articles as a reading list, and full-corpus article search — both
rendered through the **same** client-side row template, so picking an article the fiche
never cited is never the harder path (SPEC §12.3).

`GET /save` does not exist; `POST /save` **re-validates the whole candidate file** against
the committed corpus before writing anything, using the exact function
(`validate_golden_set_against_corpus`) the CLI validator runs — the helper and the
validator can never disagree about what "valid" means because they are the same code.

Run it with:

    uv run uvicorn rag.eval.annotate_app:app --reload
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

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
from rag.eval.schema import EXPECTED_STATES, GoldenItem, dump_golden_set, load_golden_set
from rag.eval.validate import GoldenSetValidationError, validate_golden_set_against_corpus

__all__ = ["create_app"]

REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES_DIR = Path(__file__).parent / "templates"


class HistoryTurnIn(BaseModel):
    role: str
    content: str


class SaveRequest(BaseModel):
    question: str
    expected_state: str
    gold_fiches: list[str] = []
    gold_articles: list[str] = []
    gold_spans: list[str] = []
    expected_points: list[str] = []
    tags: list[str] = []
    history: list[HistoryTurnIn] = []


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
                "expected_states": sorted(EXPECTED_STATES),
                "sections_json": _safe_json(sections_json),
            },
        )

    @app.post("/save")
    def save(payload: SaveRequest) -> JSONResponse:
        existing = load_golden_set(golden_set_path)
        new_id = allocate_id(id_counter_path, existing_ids=[item.id for item in existing])
        item = GoldenItem(
            id=new_id,
            question=payload.question,
            history=tuple({"role": turn.role, "content": turn.content} for turn in payload.history),
            expected_state=payload.expected_state,
            gold_fiches=tuple(payload.gold_fiches),
            gold_spans=tuple(payload.gold_spans),
            gold_articles=tuple(payload.gold_articles),
            expected_points=tuple(payload.expected_points),
            tags=tuple(payload.tags),
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

"""The demo app — FastAPI + Jinja/HTMX, one container (SPEC §13, #50).

The app **adds no logic of its own**: every answer comes from `rag.pipeline.run_chain`, the
same function the generation eval runs, behind the `AnswerFn` seam. What it adds is
presentation (`rag.app.view`), the licence surface (`rag.app.attribution`) and the HTTP
contract:

- `POST /api/ask` — the envelope as JSON, documented at `/docs` (SPEC §13.3: the typed
  contract, inspectable by curl).
- `POST /ask` — the HTML fragment HTMX appends to the conversation.

**Stateless** (SPEC §13.4): no session, no storage. Each rendered exchange carries itself
back as hidden `history_role`/`history_content` inputs, which the next `/ask` posts with
the question; `run_chain` trims that untrusted history server-side before any stage sees
it. No token streaming (SPEC §13.1): a strict `json_schema` response is unreadable until
the object closes, so the answer arrives whole.

Run it with:

    uv run uvicorn rag.app.main:create_app --factory
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from qdrant_client import QdrantClient

from rag.app.attribution import CorpusAttribution, load_attribution
from rag.app.view import build_view
from rag.condensation.chain import make_condense_fn
from rag.condensation.prompt import MAX_HISTORY_TURN_CHARS, HistoryTurn
from rag.config import Settings, load_settings
from rag.generation.chain import make_generate_fn
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import Envelope
from rag.pipeline import run_chain
from rag.retrieval.lookup import load_lookup_keys

__all__ = ["AnswerFn", "AskRequest", "create_app", "make_answer_fn"]

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).parent

# `(question, history)` in, the pipeline's own result out. The real one is `run_chain` over
# real Qdrant, BGE-M3 and OpenRouter (`make_answer_fn`); tests inject a fake, the same seam
# `GenerateFn`/`CondenseFn` already are one level down.
AnswerFn = Callable[[str, Sequence[HistoryTurn]], GenerationResult]

# The question becomes a history turn on the next request, where `trim_history` clamps it
# to this anyway — so accepting more now would only mean answering a question the
# follow-up can no longer see in full.
MAX_QUESTION_CHARS = MAX_HISTORY_TURN_CHARS

# The model writes light Markdown in `explanation` (bold, lists). `html: False` escapes any
# raw HTML it writes instead of passing it through, and markdown-it's own link validation
# drops `javascript:`-style targets — model output is rendered, never trusted as markup.
_MARKDOWN = MarkdownIt("commonmark", {"html": False})

_UNAVAILABLE = "Le service est momentanément indisponible. Réessayez dans un instant."


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    """One turn of the conversation. `history` is the client's own copy of prior turns —
    user questions and assistant explanations only, never a citation list (SPEC §10.6)."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list)


_TURNS = TypeAdapter(list[Turn])


def make_answer_fn(settings: Settings) -> AnswerFn:
    """`run_chain` over the real stores and models. The lookup-key set loads on the first
    question rather than at startup, so the app can come up before Qdrant answers (the two
    wake together — SPEC §14.2); BGE-M3 likewise loads on its first embed."""
    client = QdrantClient(settings.qdrant_url)
    condense_fn = make_condense_fn(settings)
    generate_fn = make_generate_fn(settings)

    @cache
    def lookup_keys() -> frozenset[str]:
        return load_lookup_keys(client)

    def answer(question: str, history: Sequence[HistoryTurn]) -> GenerationResult:
        from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

        return run_chain(
            question,
            history,
            client=client,
            embed=embed_batch,
            lookup_keys=lookup_keys(),
            condense_fn=condense_fn,
            generate_fn=generate_fn,
        )

    return answer


def create_app(
    *,
    answer: AnswerFn | None = None,
    attribution: CorpusAttribution | None = None,
) -> FastAPI:
    """Both arguments default to the real thing — `make_answer_fn(load_settings())` and the
    committed `corpus_manifest.json`."""
    if answer is None:
        answer = make_answer_fn(load_settings())
    if attribution is None:
        attribution = load_attribution()

    app = FastAPI(
        title="rag-assurances",
        description=(
            "Questions de droit des assurances, réponses citant le Code des assurances. "
            "Information, pas conseil."
        ),
    )
    app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=_APP_DIR / "templates")
    templates.env.globals["attribution"] = attribution
    templates.env.filters["markdown"] = lambda text: Markup(_MARKDOWN.render(text))

    def _run(question: str, history: Sequence[Turn]) -> GenerationResult | None:
        turns = [HistoryTurn(role=t.role, content=t.content) for t in history]
        try:
            return answer(question, turns)
        except Exception:
            # OpenRouter, Qdrant or the embedder — no fixed type, and none of it is the
            # client's fault. Logged in full, surfaced as one plain 503.
            logger.exception("pipeline failed for one question")
            return None

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "index.html", {"max_question_chars": MAX_QUESTION_CHARS}
        )

    @app.post("/api/ask", response_model=Envelope)
    def api_ask(body: AskRequest) -> Envelope:
        """Ask one question; get the typed answer envelope back — the same one the page
        renders, with any citation outside the retrieved context already dropped."""
        result = _run(body.question, body.history)
        if result is None:
            raise HTTPException(status_code=503, detail=_UNAVAILABLE)
        return build_view(result).envelope

    @app.post("/ask", response_class=HTMLResponse, include_in_schema=False)
    def ask(
        request: Request,
        question: Annotated[str, Form(min_length=1, max_length=MAX_QUESTION_CHARS)],
        history_role: Annotated[list[str], Form()] = [],  # noqa: B006 — FastAPI copies defaults
        history_content: Annotated[list[str], Form()] = [],  # noqa: B006
    ) -> HTMLResponse:
        if len(history_role) != len(history_content):
            raise HTTPException(status_code=422, detail="history roles and contents differ in length")
        try:
            history = _TURNS.validate_python(
                [{"role": r, "content": c} for r, c in zip(history_role, history_content, strict=True)]
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="malformed history") from exc

        result = _run(question, history)
        if result is None:
            return templates.TemplateResponse(
                request,
                "_error.html",
                {"question": question, "message": _UNAVAILABLE},
                status_code=503,
            )
        return templates.TemplateResponse(
            request, "_exchange.html", {"question": question, "view": build_view(result)}
        )

    return app

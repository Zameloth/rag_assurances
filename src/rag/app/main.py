"""The demo app — FastAPI + Jinja/HTMX, one container (SPEC §13, #50).

The app **adds no logic of its own**: every answer comes from `rag.pipeline.run_chain`, the
same function the generation eval runs, behind the `AnswerFn` seam. What it adds is
presentation (`rag.app.view`), the licence surface (`rag.app.attribution`) and the HTTP
contract:

- `POST /api/ask` — the envelope as JSON, documented at `/docs` (SPEC §13.3: the typed
  contract, inspectable by curl).
- `POST /ask` — the HTML fragment for the conversation, whole, in one response.
- `POST /ask/stream` — the same fragment as the last event of an SSE stream whose earlier
  events are the pipeline's own **stage events** (SPEC §13.2, #51). What the page uses.

**Stateless** (SPEC §13.4): no session, no storage. Each rendered exchange carries itself
back as hidden `history_role`/`history_content` inputs, which the next `/ask` posts with
the question; `run_chain` trims that untrusted history server-side before any stage sees
it. No token streaming (SPEC §13.1): a strict `json_schema` response is unreadable until
the object closes, so the answer arrives whole — the stream carries progress, never tokens,
and the connection *is* the request, so there is no job store to poll either.

Run it with:

    uv run uvicorn rag.app.main:create_app --factory
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from qdrant_client import QdrantClient

from rag.app.attribution import CorpusAttribution, load_attribution
from rag.app.health import AliasProbe, check_health, qdrant_aliases
from rag.app.view import build_view
from rag.condensation.chain import make_condense_fn
from rag.condensation.prompt import MAX_HISTORY_TURN_CHARS, HistoryTurn
from rag.config import Settings, load_settings
from rag.generation.chain import make_generate_fn
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import Envelope
from rag.pipeline import Stage, StageFn, ignore_stage, run_chain
from rag.publish import load_index_lock
from rag.retrieval.lookup import load_lookup_keys

__all__ = ["AnswerFn", "AskRequest", "create_app", "make_answer_fn"]

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).parent

# `(question, history, on_stage)` in, the pipeline's own result out. The real one is
# `run_chain` over real Qdrant, BGE-M3 and OpenRouter (`make_answer_fn`); tests inject a
# fake, the same seam `GenerateFn`/`CondenseFn` already are one level down. `on_stage` hears
# each stage as it starts — the routes that don't stream pass one that ignores it.
AnswerFn = Callable[[str, Sequence[HistoryTurn], StageFn], GenerationResult]

# The question becomes a history turn on the next request, where `trim_history` clamps it
# to this anyway — so accepting more now would only mean answering a question the
# follow-up can no longer see in full.
MAX_QUESTION_CHARS = MAX_HISTORY_TURN_CHARS

# The model writes light Markdown in `explanation` (bold, lists). `html: False` escapes any
# raw HTML it writes instead of passing it through, and markdown-it's own link validation
# drops `javascript:`-style targets — model output is rendered, never trusted as markup.
_MARKDOWN = MarkdownIt("commonmark", {"html": False})

_UNAVAILABLE = "Le service est momentanément indisponible. Réessayez dans un instant."

# The `/health` probe's own Qdrant timeout — well inside the container healthcheck's 5 s, so
# a slow store reads as "unreachable" in the JSON rather than as a probe that timed out.
HEALTH_QDRANT_TIMEOUT_SECONDS = 2

# A slow stage — generation above all — is silent until it ends; an SSE comment this often keeps any proxy
# between here and the browser from reading that silence as a dead connection.
HEARTBEAT_SECONDS = 15.0

_STAGE_LABELS = {
    Stage.CHARGEMENT: "Chargement",
    Stage.CONDENSATION: "Condensation",
    Stage.RECHERCHE: "Recherche",
    Stage.GENERATION: "Génération",
}


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    """One turn of the conversation. `history` is the client's own copy of prior turns —
    user questions and assistant explanations only, never a citation list (SPEC §10.6)."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list)


_TURNS = TypeAdapter(list[Turn])


def _form_history(
    history_role: Annotated[list[str], Form()] = [],  # noqa: B006 — FastAPI copies defaults
    history_content: Annotated[list[str], Form()] = [],  # noqa: B006
) -> list[Turn]:
    """The history an HTML form posts back — the hidden inputs each rendered exchange
    carries (SPEC §13.4), paired up and validated like `/api/ask`'s JSON history."""
    if len(history_role) != len(history_content):
        raise HTTPException(status_code=422, detail="history roles and contents differ in length")
    try:
        return _TURNS.validate_python(
            [{"role": r, "content": c} for r, c in zip(history_role, history_content, strict=True)]
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="malformed history") from exc


def load_embedder() -> None:
    from rag.ingest.embedder import load_model  # deferred: pulls in torch

    load_model()


def make_answer_fn(settings: Settings) -> AnswerFn:
    """`run_chain` over the real stores and models. BGE-M3 is already resident — the app
    loads it at startup (`create_app`) — but the lookup-key set is read from Qdrant on the
    first question rather than at startup, so the app can come up before Qdrant answers
    (the two wake together — SPEC §14.2). That read is the `chargement` stage, announced
    only while it actually runs; one that fails is retried, and announced again, on the
    next question. It holds a lock, so concurrent first questions read it once."""
    client = QdrantClient(settings.qdrant_url)
    condense_fn = make_condense_fn(settings)
    generate_fn = make_generate_fn(settings)
    warm_up_lock = threading.Lock()
    loaded_lookup_keys: frozenset[str] | None = None

    def warm_up(on_stage: StageFn) -> frozenset[str]:
        nonlocal loaded_lookup_keys
        if loaded_lookup_keys is None:
            on_stage(Stage.CHARGEMENT)
            with warm_up_lock:
                if loaded_lookup_keys is None:
                    loaded_lookup_keys = load_lookup_keys(client)
        return loaded_lookup_keys

    def answer(question: str, history: Sequence[HistoryTurn], on_stage: StageFn) -> GenerationResult:
        from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

        lookup_keys = warm_up(on_stage)
        return run_chain(
            question,
            history,
            client=client,
            embed=embed_batch,
            lookup_keys=lookup_keys,
            condense_fn=condense_fn,
            generate_fn=generate_fn,
            on_stage=on_stage,
        )

    return answer


def _nothing_to_load() -> None:
    pass


@dataclass(frozen=True)
class _Finished:
    result: GenerationResult | None


def _sse(event: str, data: str) -> str:
    # One `data:` line per line of payload: a bare newline would end the event early.
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"


def create_app(
    *,
    answer: AnswerFn | None = None,
    attribution: CorpusAttribution | None = None,
    load_models: Callable[[], None] | None = None,
    aliases: AliasProbe | None = None,
    release_tag: str | None = None,
) -> FastAPI:
    """Every argument defaults to the real thing — `make_answer_fn(load_settings())`, the
    committed `corpus_manifest.json`, the Qdrant aliases and the committed `index_lock.json`.

    `load_models` defaults to loading BGE-M3 only when `answer` does too: the models are
    the real answer function's, and a fake one has none to load.

    **Models load eagerly at startup and are never unloaded** (SPEC §14.2): the container's
    lifetime *is* the model's lifetime — no idle timer, no unload path. Startup blocks on
    the load, so a load that fails fails the process, and Sablier starts a fresh container
    on the next visit instead of one sitting unhealthy for good. The healthcheck's
    `start_period`/`retries` are what tolerate the wait.
    """
    if load_models is None:
        load_models = load_embedder if answer is None else _nothing_to_load
    if answer is None or aliases is None:
        settings = load_settings()
        if answer is None:
            answer = make_answer_fn(settings)
        if aliases is None:
            aliases = qdrant_aliases(QdrantClient(settings.qdrant_url, timeout=HEALTH_QDRANT_TIMEOUT_SECONDS))
    if attribution is None:
        attribution = load_attribution()
    if release_tag is None:
        release_tag = load_index_lock().release_tag

    models_loaded = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        load_models()
        models_loaded.set()
        yield

    app = FastAPI(
        lifespan=lifespan,
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

    def _run(
        question: str, history: Sequence[Turn], on_stage: StageFn = ignore_stage
    ) -> GenerationResult | None:
        turns = [HistoryTurn(role=t.role, content=t.content) for t in history]
        try:
            return answer(question, turns, on_stage)
        except Exception:
            # OpenRouter, Qdrant or the embedder — no fixed type, and none of it is the
            # client's fault. Logged in full, surfaced as one plain 503.
            logger.exception("pipeline failed for one question")
            return None

    def _partial(question: str, result: GenerationResult | None) -> tuple[str, dict[str, object]]:
        """The fragment one answered question renders as, on either `/ask` route."""
        if result is None:
            return "_error.html", {"question": question, "message": _UNAVAILABLE}
        return "_exchange.html", {"question": question, "view": build_view(result)}

    @app.get("/health")
    def health() -> JSONResponse:
        """SPEC §14.3: 200 only when models are loaded, Qdrant answers, and the aliases
        serve the release `index_lock.json` names — 503 otherwise, with each clause."""
        report = check_health(
            models_loaded=models_loaded.is_set(), aliases=aliases, release_tag=release_tag
        )
        return JSONResponse(report.as_dict(), status_code=200 if report.healthy else 503)

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
        history: Annotated[list[Turn], Depends(_form_history)],
    ) -> HTMLResponse:
        result = _run(question, history)
        template, context = _partial(question, result)
        status_code = 503 if result is None else 200
        return templates.TemplateResponse(request, template, context, status_code=status_code)

    @app.post("/ask/stream", include_in_schema=False)
    def ask_stream(
        question: Annotated[str, Form(min_length=1, max_length=MAX_QUESTION_CHARS)],
        history: Annotated[list[Turn], Depends(_form_history)],
    ) -> StreamingResponse:
        """SPEC §13.2: one `stage` event per stage the pipeline announces, as it announces
        it, then the rendered exchange as `result` — or the error partial as `error`. The
        input is validated before the stream opens, so a bad request is still a plain 422."""
        events: queue.Queue[Stage | _Finished] = queue.Queue()

        def work() -> None:
            events.put(_Finished(_run(question, history, events.put)))

        def stream() -> Iterator[str]:
            threading.Thread(target=work, daemon=True).start()
            while True:
                try:
                    event = events.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if isinstance(event, Stage):
                    payload = {"stage": event.value, "label": _STAGE_LABELS[event]}
                    yield _sse("stage", json.dumps(payload, ensure_ascii=False))
                else:
                    template, context = _partial(question, event.result)
                    html = templates.get_template(template).render(context)
                    yield _sse("error" if event.result is None else "result", html)
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            # `no-cache` per the SSE spec; `X-Accel-Buffering` stops a buffering proxy from
            # holding every stage until the answer, which would make the indicator a lie.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app

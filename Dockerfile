# The app image (SPEC §14) — and, run as `python -m rag.restore`, the restore tool
# (SPEC §15.5): no repo checkout, no Python and no qdrant-client on the host.
#
# The compose topology, Traefik and Sablier labels, mem_limit and the model-cache volume
# are the deploy ticket's (#54). What lives here is what travels with the image itself:
# the code, the `index_lock.json` it was committed against, the corpus manifest the
# licence footer renders, and the healthcheck.

FROM python:3.13-slim

# Pinned, not :latest — the same reason uv.lock is committed.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, in their own layer: a code change should not re-download torch.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY index_lock.json ./
COPY data/corpus/corpus_manifest.json data/corpus/LICENSE.md ./data/corpus/
# Installed in place, not as a wheel: the package finds `index_lock.json`, the corpus
# manifest and its `data/raw/hf_cache` model cache relative to the checkout root.
RUN uv sync --frozen --no-dev

EXPOSE 8000

# SPEC §14.3 — tolerating a long model load: startup blocks on BGE-M3, so for the first
# minutes the probe meets a closed port. urllib raises on a 503, so any /health clause
# failing fails the probe; the 4 s client timeout sits inside the 5 s one.
HEALTHCHECK --interval=10s --timeout=5s --retries=30 --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "rag.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

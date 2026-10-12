# Entry points, with the three pipeline target names fixed by SPEC §16.1.
#
# The ones still stubbed wait for their ticket, but they are named now and on purpose:
# the runbook in SPEC §15.3 already spells them, so a later ticket that invented its own
# name would leave the spec describing commands that do not exist. A stub exits non-zero
# rather than succeeding quietly — a no-op that prints nothing is indistinguishable from a
# run that worked.
#
# There is no `deploy`: the host clones this repo and runs the sequence itself (SPEC §15.7).

.DEFAULT_GOAL := help
.PHONY: help install test typecheck lint check up down logs clean ingest ladder publish-index

# $(call todo,<build-order step>,<what it will run>)
define todo
	@echo "make $@ is not implemented yet — build order step $(1)."
	@echo "It will run: $(2)"
	@exit 1
endef

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- development -------------------------------------------------------------

# The dev store's own file: plain `docker compose` would pick up `compose.yml`, the prod one.
DEV_COMPOSE := docker compose -f compose.dev.yml

install: ## Create the venv and install the package with its dev group
	uv sync

test: ## Run the test suite with coverage
	uv run pytest

typecheck: ## Type-check src/ and tests/
	uv run mypy

lint: ## Lint src/, tests/ and scripts/ with ruff
	uv run ruff check .

check: typecheck lint test ## Everything CI would run

up: ## Start the dev Qdrant (reachable at QDRANT_URL)
	$(DEV_COMPOSE) up -d --wait

down: ## Stop the dev Qdrant, keeping its volume
	$(DEV_COMPOSE) down

logs: ## Follow the dev Qdrant logs
	$(DEV_COMPOSE) logs -f

clean: ## Stop the dev Qdrant and delete its volume
	$(DEV_COMPOSE) down -v

# --- pipeline ----------------------------------------------------------------

ingest: ## Corpus -> chunks -> BGE-M3 -> Qdrant (SPEC §4-§7)
	uv run python -m rag.ingest

ladder: ## Run the six-rung retrieval ablation ladder (SPEC §12.8)
	$(call todo,7,the eval harness — eval/ — writing per-rung scores to eval/runs/)

# RUN names the eval/runs/<run-id>.json that scored the index being shipped — no default,
# because which run chose the deployed arm is a decision (ADR-0024), not something to infer.
publish-index: ## Dump the scored index -> GitHub Release -> index_lock.json (RUN=eval/runs/<id>.json)
	@test -n "$(RUN)" || { echo "usage: make publish-index RUN=eval/runs/<run-id>.json [TAG=index-YYYY-MM-DD]"; exit 1; }
	uv run python -m rag.publish --run $(RUN) $(if $(TAG),--tag $(TAG))


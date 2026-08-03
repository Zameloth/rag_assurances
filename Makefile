# Entry points, with the four pipeline target names fixed by SPEC §16.1.
#
# The four are stubbed until their ticket lands, but they are named now and on purpose:
# the runbooks in SPEC §15.3 and §15.7 already spell them, so a later ticket that
# invented its own name would leave the spec describing commands that do not exist.
# A stub exits non-zero rather than succeeding quietly — a no-op `make deploy` that
# prints nothing is indistinguishable from a deploy that worked.

.DEFAULT_GOAL := help
.PHONY: help install test typecheck check up down logs clean ingest ladder publish-index deploy

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

install: ## Create the venv and install the package with its dev group
	uv sync

test: ## Run the test suite
	uv run pytest

typecheck: ## Type-check src/ and tests/
	uv run mypy

check: typecheck test ## Everything CI would run

up: ## Start the dev Qdrant (reachable at QDRANT_URL)
	docker compose up -d --wait

down: ## Stop the dev Qdrant, keeping its volume
	docker compose down

logs: ## Follow the dev Qdrant logs
	docker compose logs -f

clean: ## Stop the dev Qdrant and delete its volume
	docker compose down -v

# --- pipeline ----------------------------------------------------------------

ingest: ## Corpus -> chunks -> BGE-M3 -> Qdrant (SPEC §4-§7)
	$(call todo,2,the ingest library over data/corpus — src/rag/ingest/)

ladder: ## Run the six-rung retrieval ablation ladder (SPEC §12.8)
	$(call todo,7,the eval harness — eval/ — writing per-rung scores to eval/runs/)

publish-index: ## Parquet points dump -> index_lock.json -> GitHub Release (SPEC §15.3)
	$(call todo,12,the points dump the ladder already scored -> index_lock.json -> gh release create)

deploy: ## Pull, restore, up, verify (SPEC §15.7)
	$(call todo,12,compose pull -> compose run --rm rag-assurances python -m rag.restore \
	-> compose up -d -> curl -f https://rag.theo-eloy.fr/health)

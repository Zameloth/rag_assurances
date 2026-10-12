"""`rag.eval.floor_calibration_draft` — the floor-calibration drafter (#95, #84, ADR-0020).

The LLM round trips are the existing annotation-helper propose functions, replaced here by
canned fakes: no network call, no spend in a test run.
"""

from __future__ import annotations

import subprocess
import sys

# Modules whose presence in `sys.modules` after importing the drafter would mean an import
# path reaches the pipeline under test (the retrieval package, its Qdrant client).
_FORBIDDEN_PREFIXES = ("rag.retrieval", "qdrant_client")


def test_importing_the_drafter_never_loads_the_retrieval_pipeline() -> None:
    """ADR-0020's central constraint, checked on the *transitive* import graph — a fresh
    interpreter imports the drafter, then we look at what got loaded. Gold labels chosen by
    the reranker would calibrate the reranker against itself."""
    probe = (
        "import sys, rag.eval.floor_calibration_draft;"
        f"bad = sorted(m for m in sys.modules if m.startswith({_FORBIDDEN_PREFIXES!r}));"
        "print(bad)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"

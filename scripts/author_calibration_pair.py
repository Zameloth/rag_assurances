#!/usr/bin/env python3
"""Author one calibration pair: a real pipeline answer and its hand-made faulted twin
(SPEC §12.10, #47).

    uv run python scripts/author_calibration_pair.py gs-014
    uv run python scripts/author_calibration_pair.py gs-014 --archetype dropped_expected_point

Runs the full chain (condensation when the item has history, retrieval on the
ladder-winning arm, generation) for the golden item, prints the answer next to the four
fault archetypes, then opens `$EDITOR` on a copy of the answer for you to plant **one**
fault in. You label both answers pass/fail yourself — the real answer is not assumed clean. The pair is validated against the golden set and each archetype's structural
rules before it is written to `eval/calibration/judge-set.yaml`; an invalid edit reopens
the editor with the reason, and nothing is written until it passes. An existing pair for
the same golden item is replaced.

Needs Qdrant up (`make up`) and `OPENROUTER_API_KEY`; costs one or two OpenRouter calls.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from qdrant_client import QdrantClient

from rag.condensation.chain import make_condense_fn
from rag.condensation.prompt import HistoryTurn
from rag.config import load_settings
from rag.eval.calibration import CalibrationAnswer, CalibrationSetError, FaultArchetype, HumanLabel
from rag.eval.calibration_authoring import (
    ARCHETYPE_GUIDANCE,
    draft_pair,
    envelope_from_yaml,
    envelope_to_yaml,
    parse_label,
    upsert_pair,
)
from rag.eval.judge import render_answer
from rag.eval.paths import CALIBRATION_SET_PATH, GOLDEN_SET_PATH, REPO_ROOT
from rag.eval.run_generation_experiment import GENERATION_RETRIEVAL_ARM, run_chain
from rag.eval.schema import load_golden_set
from rag.generation.chain import make_generate_fn
from rag.retrieval.lookup import load_lookup_keys


def _choose_archetype() -> FaultArchetype:
    archetypes = list(FaultArchetype)
    while True:
        raw = input(f"archetype [1-{len(archetypes)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(archetypes):
            return archetypes[int(raw) - 1]
        print("pick one of the numbers above")


def _ask_label(question: str, default: HumanLabel) -> HumanLabel:
    while True:
        label = parse_label(input(f"{question} [pass/fail, Enter = {default.value}]: "), default=default)
        if label is not None:
            return label
        print("answer pass or fail")


def _edit(text: str) -> str:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        subprocess.run([*shlex.split(editor), str(path)], check=True)
        return path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("golden_id", help="the golden item to draw the pair from, e.g. gs-014")
    parser.add_argument("--archetype", choices=[a.value for a in FaultArchetype], help="skip the prompt")
    args = parser.parse_args(argv)

    golden_set = load_golden_set(GOLDEN_SET_PATH)
    item = next((i for i in golden_set if i.id == args.golden_id), None)
    if item is None:
        print(f"{args.golden_id}: not in {GOLDEN_SET_PATH}", file=sys.stderr)
        return 1

    settings = load_settings()
    client = QdrantClient(settings.qdrant_url)
    from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

    try:
        result = run_chain(
            item.question,
            tuple(HistoryTurn(role=turn["role"], content=turn["content"]) for turn in item.history),  # type: ignore[arg-type]
            client=client,
            embed=embed_batch,
            lookup_keys=load_lookup_keys(client),
            condense_fn=make_condense_fn(settings),
            generate_fn=make_generate_fn(settings),
            retrieval_arm=GENERATION_RETRIEVAL_ARM,
        )
    finally:
        client.close()

    print(f"\n{item.id} — {item.question}  (expected: {item.expected_state})")
    print("expected points:")
    for index, point in enumerate(item.expected_points, start=1):
        print(f"  {index}. {point}")
    print("\n--- clean answer (the real pipeline output) ---")
    print(render_answer(result.envelope))
    print("\n--- fault archetypes ---")
    for index, archetype in enumerate(FaultArchetype, start=1):
        print(f"  {index}. {archetype.value}\n     {ARCHETYPE_GUIDANCE[archetype]}")

    archetype = FaultArchetype(args.archetype) if args.archetype else _choose_archetype()
    pair = draft_pair(item, result, archetype)
    # SPEC §12.10: every answer human-labelled — the real answer is not assumed clean.
    clean_label = _ask_label("\nyour label for the clean (real) answer above", HumanLabel.PASS)
    pair = replace(pair, clean=CalibrationAnswer(pair.clean.envelope, clean_label))
    print(f"\nretrieved_citation_ids: {', '.join(pair.retrieved_citation_ids) or '(none)'}")
    input(f"\n{archetype.value}: {ARCHETYPE_GUIDANCE[archetype]}\npress Enter to open the editor on the twin ")

    header = f"# {archetype.value}: {ARCHETYPE_GUIDANCE[archetype]}\n# Plant exactly one fault.\n"
    draft = header + envelope_to_yaml(pair.faulted.envelope)
    twin_label: HumanLabel | None = None
    while True:
        draft = _edit(draft)
        try:
            envelope = envelope_from_yaml(draft)
            if twin_label is None:
                print("\n--- faulted twin ---")
                print(render_answer(envelope))
                twin_label = _ask_label("\nyour label for the faulted twin", HumanLabel.FAIL)
            upsert_pair(CALIBRATION_SET_PATH, pair.with_faulted(CalibrationAnswer(envelope, twin_label)), golden_set)
        except CalibrationSetError as error:
            print(f"\nnot written:\n{error}")
            if input("edit again? [Y/n] ").strip().lower() == "n":
                return 1
            continue
        break

    print(f"\nwritten to {CALIBRATION_SET_PATH.relative_to(REPO_ROOT)} ({item.id}, {archetype.value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

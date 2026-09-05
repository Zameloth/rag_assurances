"""`allocate_id` — golden-set ids assigned once, never reused (SPEC §12.1, #33)."""

from pathlib import Path

from rag.eval.ids import allocate_id


def test_first_id_with_no_counter_and_no_existing_ids(tmp_path: Path) -> None:
    counter_path = tmp_path / ".next_id"
    assert allocate_id(counter_path, existing_ids=[]) == "gs-001"


def test_bootstraps_from_the_highest_existing_id(tmp_path: Path) -> None:
    counter_path = tmp_path / ".next_id"
    assert allocate_id(counter_path, existing_ids=["gs-001", "gs-014", "gs-003"]) == "gs-015"


def test_increments_across_calls(tmp_path: Path) -> None:
    counter_path = tmp_path / ".next_id"
    first = allocate_id(counter_path, existing_ids=[])
    second = allocate_id(counter_path, existing_ids=[])
    third = allocate_id(counter_path, existing_ids=[])
    assert [first, second, third] == ["gs-001", "gs-002", "gs-003"]


def test_never_reuses_an_id_even_after_deletion(tmp_path: Path) -> None:
    """The counter file, not the current file's contents, is authoritative — deleting
    gs-002 from the golden set must not let a later save hand it out again."""
    counter_path = tmp_path / ".next_id"
    allocate_id(counter_path, existing_ids=[])  # gs-001
    allocate_id(counter_path, existing_ids=["gs-001"])  # gs-002
    # gs-002 is deleted from the golden set between saves — existing_ids no longer has it.
    third = allocate_id(counter_path, existing_ids=["gs-001"])
    assert third == "gs-003"


def test_survives_a_counter_file_older_than_the_golden_set(tmp_path: Path) -> None:
    """If the counter file is missing or behind (e.g. hand-edited golden set), the high
    water mark is whichever of the two is further along."""
    counter_path = tmp_path / ".next_id"
    counter_path.write_text("2")
    assert allocate_id(counter_path, existing_ids=["gs-014"]) == "gs-015"

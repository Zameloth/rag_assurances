"""SPEC §4.2 / ADR-0003 — texteHtml block extraction and table stripping."""

from rag.ingest.html_blocks import extract_blocks, extract_text


def test_extract_text_reproduces_texte_for_inline_links() -> None:
    """A link's text picks up a space on both sides, matching how `texte` was built."""
    html = "<p>Elle a été fixée en application de l'<a href=\"x\">article L. 112-2</a>.</p>"
    assert extract_text(html) == "Elle a été fixée en application de l' article L. 112-2 ."


def test_extract_text_strips_table_content() -> None:
    html = "<p>Before.</p><table><tbody><tr><td><p>1</p></td><td><p>2</p></td></tr></tbody></table><p>After.</p>"
    assert extract_text(html) == "Before. After."


def test_extract_blocks_splits_on_boundary_tags() -> None:
    html = "<p>First.</p><p>Second.</p>"
    assert extract_blocks(html) == ["First.", "Second."]


def test_extract_blocks_handles_nested_p() -> None:
    """DILA's markup is not well-formed — `<p>` nests inside `<p>` (see a live sample)."""
    html = "<p>Outer lead-in</p><p><p>Inner text</p></p><p>Next</p>"
    assert extract_blocks(html) == ["Outer lead-in", "Inner text", "Next"]


def test_extract_blocks_drops_empty_blocks() -> None:
    html = "<p></p><p>Only real block</p>"
    assert extract_blocks(html) == ["Only real block"]


def test_extract_blocks_excludes_table_rows() -> None:
    """`<tr>` is a boundary tag in principle, but every `<tr>` in this corpus sits inside a
    `<table>`, which is dropped wholesale before the boundary check runs."""
    html = "<p>Prose.</p><table><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>"
    assert extract_blocks(html) == ["Prose."]


def test_extract_blocks_splits_on_br_and_li() -> None:
    html = "<p>Line one<br>Line two</p><ul><li>Item A</li><li>Item B</li></ul>"
    assert extract_blocks(html) == ["Line one", "Line two", "Item A", "Item B"]

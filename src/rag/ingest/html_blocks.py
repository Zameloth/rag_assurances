"""`texteHtml` → text and blocks, with `<table>` content stripped (SPEC §4.2, ADR-0003).

`texte` carries zero newlines corpus-wide, so paragraph structure exists only in
`texteHtml` — this is the sole basis for splitting the 12% of over-band articles (SPEC §4).
The DILA markup is not well-formed (`<p>` nests inside `<p>`), so parsing uses the stdlib's
forgiving `HTMLParser` rather than an XML tree.

Text reconstruction joins every text node with a single space and collapses whitespace —
this, not raw concatenation, is what gets extracted text within a hair of `texte` (SPEC
§3.2's "identical, difflib 1.000" claim — measured 0.98+ on every one of the 2,362
table-free articles, exact on 98% of them): `texte` was itself built by joining each node's
text with a space, so an inline element like `<a>…</a>` picks up a space on both sides even
where the surrounding markup has none. The rare misses are DILA authoring artifacts —
literal double spaces mid-sentence in the source — that whitespace collapsing normalizes
away; trading that for byte-exact reproduction would mean shipping the typo into every
chunk. This is, strictly, a normalization the #23 ticket's "no enrichment, no normalization"
line doesn't carve out — confirmed as an accepted, deliberate exception rather than a silent
deviation, on the reasoning above.

`<table>` subtrees are dropped wholesale during the same walk — every descendant tag and
text node inside a `<table>…</table>` is skipped, which is what "table content is stripped;
everything else is kept" (SPEC §4.2) means in practice.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

__all__ = ["extract_text", "extract_blocks"]

# The four tags SPEC §4.2 names as split points. `<tr>` is listed for a splitter that has
# to be general — in practice it never fires on this corpus, because every `<tr>` lives
# inside a `<table>` that is stripped before the boundary check runs.
_BOUNDARY_TAGS = frozenset({"p", "br", "li", "tr"})

_WHITESPACE_RE = re.compile(r"\s+")


def extract_text(html: str) -> str:
    """The article's plain text with `<table>` content removed, one span, no boundaries.

    Used for the whole-article-fits-the-band check and as the point `text` when it does.
    """
    parser = _BlockExtractor()
    parser.feed(html)
    parser.close()
    return _collapse(" ".join(parser.blocks))


def extract_blocks(html: str) -> list[str]:
    """The article split at `<p>`/`<br>`/`<li>`/`<tr>` boundaries, `<table>` content removed.

    Empty blocks (an empty `<p></p>`, or two boundary tags back to back) are dropped —
    they carry no text to pack and would only distort a mean-block-size assumption
    downstream.
    """
    parser = _BlockExtractor()
    parser.feed(html)
    parser.close()
    return parser.blocks


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


class _BlockExtractor(HTMLParser):
    """Streams `texteHtml` into blocks, dropping everything inside `<table>`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._current: list[str] = []
        self._table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._on_start(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._on_start(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._table_depth:
            return
        self._current.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _on_start(self, tag: str) -> None:
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth:
            return
        if tag in _BOUNDARY_TAGS:
            self._flush()

    def _flush(self) -> None:
        text = _collapse(" ".join(self._current))
        if text:
            self.blocks.append(text)
        self._current = []

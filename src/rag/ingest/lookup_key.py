"""The article-number normalizer (SPEC §7.3).

`citation_id` (DILA's raw `num`) and `lookup_key` are deliberately two fields: the first is
what the model copies into a citation and what the citation guardrail compares against, the
second is what the short-circuit (SPEC §9.1) indexes on. Conflating them would either lose
`R*113-4`'s legally meaningful asterisk or make the 21 prose annexe labels (`Annexe à
l'article A121-1`) collide with a real numbered article.

This is the **field validator** — anchored, applied to a `num` already known to be one
article's whole value. It is deliberately not the query-side scanner SPEC §9.1 also
describes: that one is unanchored with a maximal-munch guarantee and belongs to retrieval,
not ingest.
"""

from __future__ import annotations

import re

__all__ = ["normalize_lookup_key"]

# Full match only: uppercase L/R/A/D, an optional decree-en-Conseil-d'État asterisk, then
# any mix of a single space/dot before the digits, then one or more dash-joined digit
# groups. Anything not matching *in full* — the 21 prose annexe labels above all — is not
# an article number and gets `None`.
_PATTERN = re.compile(r"^[LRAD]\*?\s?\.?\s?\d+(?:-\d+)*$")

# The three characters SPEC §7.3 says lookup_key strips on top of uppercasing. `-` is
# deliberately absent — dash segments are load-bearing (`L113-15` != `L113-15-2`).
_STRIP = str.maketrans("", "", " .*")


def normalize_lookup_key(num: str) -> str | None:
    """Return the strict-normalized `lookup_key`, or `None` if `num` doesn't fully match.

    `None` is not an error case — it is the correct value for the 21 prose annexe labels,
    which stay searchable and citable via `citation_id` but never short-circuit.
    """
    candidate = num.strip()
    if not _PATTERN.match(candidate):
        return None
    return candidate.upper().translate(_STRIP)

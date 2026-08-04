"""SPEC §1.2 / #21 — fiche scope-rule parsing."""

from rag.ingest.fiches import in_scope, parse_fiche

FICHE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F2594"
             type="Fiche d'information conditionnée">
  <dc:title>Modification du contrat d'assurance habitation</dc:title>
  <dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000006157200, \
https://www.legifrance.gouv.fr/codes/id/LEGISCTA000006158221</dc:source>
</Publication>
""".encode()

FICHE_XML_NO_SOURCE = """<?xml version="1.0" encoding="UTF-8"?>
<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F9999"
             type="Fiche Question-réponse conditionnée">
  <dc:title>Sans source</dc:title>
</Publication>
""".encode()

FICHE_XML_DUPLICATE_SOURCE = b"""<?xml version="1.0" encoding="UTF-8"?>
<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1234">
  <dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000006159534, \
https://www.legifrance.gouv.fr/codes/id/LEGISCTA000006159534</dc:source>
</Publication>
"""


def test_parse_fiche_reads_id_and_legiscta_ids() -> None:
    parsed = parse_fiche(FICHE_XML)
    assert parsed.fiche_id == "F2594"
    assert parsed.section_ids == ["LEGISCTA000006157200", "LEGISCTA000006158221"]


def test_parse_fiche_handles_a_missing_dc_source() -> None:
    parsed = parse_fiche(FICHE_XML_NO_SOURCE)
    assert parsed.fiche_id == "F9999"
    assert parsed.section_ids == []


def test_parse_fiche_dedupes_repeated_ids() -> None:
    parsed = parse_fiche(FICHE_XML_DUPLICATE_SOURCE)
    assert parsed.section_ids == ["LEGISCTA000006159534"]


def test_in_scope_true_on_nonempty_intersection() -> None:
    assert in_scope(["LEGISCTA000006157200"], {"LEGISCTA000006157200", "LEGISCTA000006999999"})


def test_in_scope_false_on_empty_intersection() -> None:
    assert not in_scope(["LEGISCTA000006157200"], {"LEGISCTA000006999999"})


def test_in_scope_false_on_no_section_ids() -> None:
    assert not in_scope([], {"LEGISCTA000006157200"})

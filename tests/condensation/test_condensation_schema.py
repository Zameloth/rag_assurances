"""SPEC §8.3 — the one-field output contract."""

from __future__ import annotations

from rag.condensation.schema import CondenserOutput


def test_requete_round_trips_through_model_validate() -> None:
    output = CondenserOutput.model_validate({"requete": "Et si je suis locataire ?"})

    assert output.requete == "Et si je suis locataire ?"


def test_schema_root_is_a_plain_object_not_a_union() -> None:
    """Unlike `Envelope` (`rag.generation.schema`), `CondenserOutput` needs no container
    wrapper for `with_structured_output(strict=True)` — its JSON schema root is already a
    plain object, never an `anyOf`."""
    schema = CondenserOutput.model_json_schema()

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert list(schema["properties"]) == ["requete"]

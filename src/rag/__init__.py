"""The rag_assurances pipeline, as an importable library.

Everything the project does — ingestion, retrieval, generation, restore — is a function
of this package rather than a script, so the ladder, the app and the deploy path all run
the same code. See SPEC §16.1 for the layout and `rag.config` for how it is configured.
"""

from rag.config import ConfigurationError, Settings, load_settings

__all__ = ["ConfigurationError", "Settings", "load_settings"]

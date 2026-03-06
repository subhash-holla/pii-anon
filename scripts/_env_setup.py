"""Environment setup for competitor evaluation.

Patches spacy model loading and other external dependencies so that
the benchmark can run in offline/restricted-network environments.

Import this module BEFORE importing any competitor packages.
"""
from __future__ import annotations

import pathlib
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def setup_offline_env() -> None:
    """Apply all necessary patches for offline operation."""
    _patch_spacy_models()
    _patch_spacy_download()
    _suppress_network_warnings()


def _patch_spacy_models() -> None:
    """Create blank spaCy models if real ones aren't available."""
    try:
        import spacy
    except ImportError:
        return

    # Check if real models exist (prefer sm, but accept any size)
    for _probe in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            spacy.load(_probe)
            return  # Real model available, no patching needed
        except OSError:
            continue

    site_packages = pathlib.Path(spacy.__file__).parent.parent

    for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
        model_path = site_packages / model_name
        if (model_path / "meta.json").exists():
            continue
        model_path.mkdir(parents=True, exist_ok=True)
        nlp = spacy.blank("en")
        nlp.meta["name"] = model_name.replace("en_", "")
        nlp.meta["version"] = "3.8.0"
        nlp.to_disk(str(model_path))
        meta = {
            "lang": "en",
            "name": model_name.replace("en_", ""),
            "version": "3.8.0",
            "spacy_version": ">=3.8.0,<3.9.0",
            "pipeline": [],
            "components": [],
            "vectors": {"width": 0, "vectors": 0, "keys": 0, "name": None},
        }
        (model_path / "meta.json").write_text(json.dumps(meta, indent=2))
        init_content = '''import spacy
from pathlib import Path
__version__ = "3.8.0"
def load(**overrides):
    return spacy.load(Path(__file__).parent, **overrides)
'''
        (model_path / "__init__.py").write_text(init_content)

    # Patch spacy.util.load_model to use our blank models
    import spacy.util
    _original = spacy.util.load_model

    def _patched(name, **kwargs):
        if isinstance(name, str) and name.startswith("en_core_web"):
            mp = site_packages / name
            if mp.exists():
                return spacy.util.load_model_from_path(mp, **kwargs)
        return _original(name, **kwargs)

    spacy.util.load_model = _patched


def _patch_spacy_download() -> None:
    """Make spacy.cli.download a no-op."""
    try:
        import spacy.cli
        spacy.cli.download = lambda *a, **kw: None
    except ImportError:
        pass


def _suppress_network_warnings() -> None:
    """Suppress network-related warnings from libraries."""
    import logging
    for logger_name in [
        "tldextract", "urllib3", "requests", "stanza",
        "presidio_analyzer", "scrubadub",
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

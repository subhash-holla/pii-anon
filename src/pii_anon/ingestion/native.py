"""Native-format reader port, registry, and capability surface (S7-01, DC-14).

Implements **FR-031** (native-format readers — PDF / image / screenshot /
DICOM / audio — emit the SAME ``Iterator[IngestRecord]`` contract the text
readers emit, so the downstream pipeline is unchanged). Upholds **NFR-026**
with the LOUD flavor the requirement demands ("no silent recall loss"): a
reader whose backend is missing reports ``dependency_available=False``
truthfully and RAISES an error naming the install extra on ``read()`` — it
never silently yields empty text. **AX-001** (no real PII anywhere),
**AX-002** (deterministic reads).

The registry mirrors :class:`~pii_anon.eval_framework.rating.registry.RatingEngineRegistry`
(thread-safe dict, fail-closed ``register``, entry-point discovery under the
DISTINCT group ``pii_anon.readers`` with per-entry-point try/except — an
absent or broken third-party reader package never breaks enumeration).

Capability honesty (FR-032): NO native reader claims round-trip
reconstruction (``supports_reconstruction=False``); byte-for-byte round-trip
stays REAL for the text formats via :mod:`pii_anon.ingestion.writers`.

# SWITCH-POINT(ORCH): surfacing reader capabilities on the orchestrator's
# ``capabilities()`` requires the protected user-WIP orchestrator.py (the
# S2-03 block class) — the surface lives HERE (`reader_capabilities()`)
# until that clears. Pass-2.
# SWITCH-POINT(OCR): real OCR strength (pytesseract + Pillow) — install
# `pii-anon[ocr]`; extraction quality validation is Pass-2.
# SWITCH-POINT(DICOM): real DICOM header/pixel handling (pydicom) — install
# `pii-anon[dicom]`; Pass-2.
# SWITCH-POINT(AUDIO-ASR): no ASR backend is shipped; the audio reader
# fails loud by design until one is chosen. Pass-2.
# SWITCH-POINT(DATA): the corpus-scale per-modality recall benchmark
# (FR-034-full) is eval-data Pass-2; the representative harness + the
# FR-035 regression gate live in tests/test_native_readers.py.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from threading import Lock
from typing import Protocol, runtime_checkable

from .schema import IngestConfig, IngestRecord

__all__ = [
    "AudioReader",
    "DicomReader",
    "ImageOcrReader",
    "NativeReader",
    "NativeReaderRegistry",
    "ReaderCapabilities",
    "ScreenshotOcrReader",
    "default_reader_registry",
    "reader_capabilities",
]


@dataclass
class ReaderCapabilities:
    """Capability declaration for a native-format reader.

    Mirrors :class:`~pii_anon.types.EngineCapabilities` for the reader
    surface: what the reader extracts, whether its backend is installed,
    and whether it can reconstruct the source (FR-032 honesty — none can,
    yet).
    """

    format_name: str
    native_dependency: str | None
    dependency_available: bool
    extracts_text: bool = True
    supports_reconstruction: bool = False
    notes: str = ""


@runtime_checkable
class NativeReader(Protocol):
    """Structural contract every native reader satisfies (FR-031)."""

    format_name: str

    def capabilities(self) -> ReaderCapabilities:
        """Report what this reader can do in the current environment."""
        ...

    def read(
        self, path: str | Path, config: IngestConfig
    ) -> Iterator[IngestRecord]:
        """Yield records from *path* under the uniform IngestRecord contract."""
        ...


def _dependency_available(module_name: str) -> bool:
    """True when *module_name* is importable (never imports it)."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The registry (the rating-registry pattern, reader-flavored)
# ---------------------------------------------------------------------------

class NativeReaderRegistry:
    """Thread-safe name-keyed registry of :class:`NativeReader` instances."""

    def __init__(self) -> None:
        self._readers: dict[str, NativeReader] = {}
        self._lock = Lock()

    def register(self, name: str, reader: NativeReader) -> None:
        """Register ``reader`` under ``name`` (overwrites an existing entry).

        Fail-closed: an object that does not satisfy the
        :class:`NativeReader` protocol is refused with :class:`TypeError`.
        """
        if not isinstance(reader, NativeReader):
            raise TypeError(
                f"native reader {name!r} must satisfy the NativeReader "
                f"protocol; got {type(reader).__name__}"
            )
        with self._lock:
            self._readers[name] = reader

    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry (no-op when absent)."""
        with self._lock:
            self._readers.pop(name, None)

    def get(self, name: str) -> NativeReader | None:
        """Return the reader registered under ``name``, or ``None``."""
        with self._lock:
            return self._readers.get(name)

    def names(self) -> list[str]:
        """Return the registered reader names, sorted."""
        with self._lock:
            return sorted(self._readers.keys())

    def discover_entrypoint_readers(
        self, group: str = "pii_anon.readers"
    ) -> list[str]:
        """Discover and register readers advertised under ``group``.

        Resolves both class targets (instantiated with no args) and
        pre-built instances; anything not satisfying :class:`NativeReader`
        is skipped. Any enumeration or load failure is swallowed (NFR-026
        graceful degradation) so an absent / broken extra never raises.
        """
        discovered: list[str] = []
        try:
            entry_points = metadata.entry_points()
            selected: Iterable[metadata.EntryPoint]
            if hasattr(entry_points, "select"):
                selected = entry_points.select(group=group)
            else:
                selected = entry_points.get(group) or ()
        except Exception:
            return discovered

        for ep in selected:
            try:
                loaded = ep.load()
                instance = loaded() if isinstance(loaded, type) else loaded
                if not isinstance(instance, NativeReader):
                    continue
            except Exception:
                continue
            self.register(ep.name, instance)
            discovered.append(ep.name)
        return discovered


# ---------------------------------------------------------------------------
# Capability-honest lazy readers (loud on read, light on construction)
# ---------------------------------------------------------------------------

class ImageOcrReader:
    """Image OCR reader — lazy ``pytesseract`` + ``Pillow`` (FR-031)."""

    format_name = "image"
    native_dependency: str | None = "pytesseract"

    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            format_name=self.format_name,
            native_dependency=self.native_dependency,
            dependency_available=(
                _dependency_available("pytesseract")
                and _dependency_available("PIL")
            ),
            extracts_text=True,
            supports_reconstruction=False,
            notes="OCR text extraction; install `pii-anon[ocr]`. # SWITCH-POINT(OCR)",
        )

    def read(
        self, path: str | Path, config: IngestConfig
    ) -> Iterator[IngestRecord]:
        """OCR the image into one text record — LOUD when the extra is absent."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                f"{self.format_name} reading requires OCR support. "
                "Install with: pip install pii-anon[ocr]  "
                "(pytesseract + Pillow; plus the tesseract binary)"
            ) from exc

        with Image.open(Path(path)) as image:
            text = str(pytesseract.image_to_string(image))
        if config.max_record_chars and len(text) > config.max_record_chars:
            text = text[: config.max_record_chars]
        yield IngestRecord(
            record_id=0,
            text=text,
            metadata={
                "modality": self.format_name,
                "source_coords": {"source": str(path)},
            },
        )


class ScreenshotOcrReader(ImageOcrReader):
    """Screenshot OCR reader — the image OCR backend, screenshot-labelled.

    Extension-indistinguishable from ``image``; selected explicitly by name
    when the caller knows the artifact is a screen capture (DC-14 lists the
    modalities separately, so capability reporting stays per-modality).
    """

    format_name = "screenshot"


class DicomReader:
    """DICOM reader — lazy ``pydicom``; header text elements only (FR-031)."""

    format_name = "dicom"
    native_dependency: str | None = "pydicom"

    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            format_name=self.format_name,
            native_dependency=self.native_dependency,
            dependency_available=_dependency_available("pydicom"),
            extracts_text=True,
            supports_reconstruction=False,
            notes=(
                "DICOM header text elements (PatientName-class values); "
                "install `pii-anon[dicom]`. # SWITCH-POINT(DICOM)"
            ),
        )

    def read(
        self, path: str | Path, config: IngestConfig
    ) -> Iterator[IngestRecord]:
        """Read header text elements into one record — LOUD when absent."""
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "DICOM reading requires pydicom. "
                "Install with: pip install pii-anon[dicom]"
            ) from exc

        dataset = pydicom.dcmread(str(path))
        parts: list[str] = []
        for element in dataset:
            value = element.value
            if isinstance(value, str) and value.strip():
                keyword = element.keyword or str(element.tag)
                parts.append(f"{keyword}: {value.strip()}")
        text = "\n".join(parts)
        if config.max_record_chars and len(text) > config.max_record_chars:
            text = text[: config.max_record_chars]
        yield IngestRecord(
            record_id=0,
            text=text,
            metadata={
                "modality": self.format_name,
                "source_coords": {"source": str(path)},
            },
        )


class AudioReader:
    """Audio reader — NO ASR backend is shipped; fails LOUD by design.

    NFR-026 "no silent recall loss": an audio reader that yielded
    empty-text records would silently drop every spoken PII span. Until an
    ASR backend is chosen (# SWITCH-POINT(AUDIO-ASR), Pass-2) ``read()``
    raises, and capabilities report ``extracts_text=False`` honestly.
    """

    format_name = "audio"
    native_dependency: str | None = None

    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            format_name=self.format_name,
            native_dependency=None,
            dependency_available=False,
            extracts_text=False,
            supports_reconstruction=False,
            notes=(
                "No ASR backend shipped; read() fails loud (never silent "
                "empty text). # SWITCH-POINT(AUDIO-ASR)"
            ),
        )

    def read(
        self, path: str | Path, config: IngestConfig
    ) -> Iterator[IngestRecord]:
        """Always raises: audio transcription needs an ASR backend (Pass-2)."""
        del config
        raise NotImplementedError(
            f"audio transcription for {str(path)!r} requires an ASR backend "
            "(# SWITCH-POINT(AUDIO-ASR), Pass-2); refusing to yield silent "
            "empty-text records (NFR-026: no silent recall loss)"
        )


# ---------------------------------------------------------------------------
# Default registry + the capability surface
# ---------------------------------------------------------------------------

def default_reader_registry() -> NativeReaderRegistry:
    """Build the registry of the five in-tree native readers."""
    # Imported here (not at module level) so native_pdf can import
    # ReaderCapabilities from this module without a cycle.
    from .native_pdf import PdfTextReader

    registry = NativeReaderRegistry()
    readers: tuple[NativeReader, ...] = (
        PdfTextReader(),
        ImageOcrReader(),
        ScreenshotOcrReader(),
        DicomReader(),
        AudioReader(),
    )
    for reader in readers:
        registry.register(reader.format_name, reader)
    return registry


def reader_capabilities() -> list[ReaderCapabilities]:
    """Capability rows for the five in-tree readers, sorted by name.

    The CLI/docs surface for "what can pii-anon ingest here?" — kept OFF
    the orchestrator (# SWITCH-POINT(ORCH): user-WIP orchestrator.py).
    """
    registry = default_reader_registry()
    rows: list[ReaderCapabilities] = []
    for name in registry.names():
        reader = registry.get(name)
        if reader is not None:
            rows.append(reader.capabilities())
    return rows

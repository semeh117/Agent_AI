"""Lightweight PyPDF extraction for Agent 2 and Agent 3 CV uploads.

This module stops at local document extraction. It does not call an LLM,
identify skills, normalize concepts, or calculate compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
from pathlib import Path
import re
import unicodedata

from pydantic import BaseModel, Field

from next_chapter.storage.extraction_cache import get_cached, set_cached


_KNOWN_CV_SECTIONS = (
    "professional summary",
    "profile",
    "technical skills",
    "skills",
    "professional experience",
    "work experience",
    "experience",
    "projects",
    "education",
    "certifications",
    "languages",
)

EXTRACTION_VERSION = "agent2-pypdf-extractor-v2"


@dataclass(frozen=True)
class Agent2Document:
    """Text and metadata extracted from one CV PDF."""

    text: str
    backend: str
    source_path: str
    content_hash: str
    extraction_version: str
    detected_sections: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class _CachedAgent2Document(BaseModel):
    """Serializable extraction payload stored independently from its path."""

    text: str
    backend: str
    content_hash: str
    extraction_version: str
    detected_sections: list[str]
    warnings: list[str] = Field(default_factory=list)


def _document_from_cache(
    cached: _CachedAgent2Document,
    source_path: Path,
) -> Agent2Document:
    return Agent2Document(
        text=cached.text,
        backend=cached.backend,
        source_path=str(source_path),
        content_hash=cached.content_hash,
        extraction_version=cached.extraction_version,
        detected_sections=tuple(cached.detected_sections),
        warnings=tuple(cached.warnings),
    )


def _document_for_cache(document: Agent2Document) -> _CachedAgent2Document:
    return _CachedAgent2Document(
        text=document.text,
        backend=document.backend,
        content_hash=document.content_hash,
        extraction_version=document.extraction_version,
        detected_sections=list(document.detected_sections),
        warnings=list(document.warnings),
    )


def _normalize_text(value: str) -> str:
    """Normalize PDF text while preserving useful line boundaries."""

    text = html.unescape(unicodedata.normalize("NFKC", str(value or "")))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(
        r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])-[ \t]*(?:\n[ \t]*)+(?=[a-zà-öø-ÿ])",
        "",
        text,
    )
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def detect_cv_sections(text: str) -> tuple[str, ...]:
    """Return recognizable CV headings in their document order."""

    found: list[str] = []
    seen: set[str] = set()
    for raw_line in _normalize_text(text).splitlines():
        candidate = raw_line.strip().strip(" :-")
        key = candidate.casefold()
        if key not in _KNOWN_CV_SECTIONS or key in seen:
            continue
        found.append(candidate)
        seen.add(key)
    return tuple(found)


def _extract_pypdf_text(source_path: Path) -> str:
    """Extract selectable text from every PDF page."""

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyPDF is required for CV extraction.") from exc

    reader = PdfReader(source_path)
    return _normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))


def extract_cv_document_agent2(
    pdf_source: str | Path,
    *,
    use_cache: bool = True,
) -> Agent2Document:
    """Extract selectable text from one local CV PDF."""

    source_path = Path(pdf_source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CV PDF does not exist: {source_path}")
    if source_path.suffix.casefold() != ".pdf":
        raise ValueError(f"CV extraction expects a PDF: {source_path}")

    content_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    cache_identity = f"{EXTRACTION_VERSION}:{content_hash}"
    if use_cache:
        cached = get_cached("agent2_document", cache_identity, _CachedAgent2Document)
        if (
            cached is not None
            and cached.content_hash == content_hash
            and cached.extraction_version == EXTRACTION_VERSION
        ):
            print("  [CACHE HIT] CV document already extracted — PyPDF skipped.")
            return _document_from_cache(cached, source_path)

    text = _extract_pypdf_text(source_path)
    if not text:
        raise ValueError(
            "This PDF contains no selectable text. Export the CV directly from "
            "Word, Google Docs, or Canva instead of uploading a scanned image."
        )

    sections = detect_cv_sections(text)
    warnings: list[str] = []
    if not sections:
        warnings.append("No recognizable CV section headings were detected.")
    if len(text) < 200:
        warnings.append(f"The extracted CV is unusually short ({len(text)} characters).")

    document = Agent2Document(
        text=text,
        backend="pypdf",
        source_path=str(source_path),
        content_hash=content_hash,
        extraction_version=EXTRACTION_VERSION,
        detected_sections=sections,
        warnings=tuple(warnings),
    )
    if use_cache:
        set_cached("agent2_document", cache_identity, _document_for_cache(document))
    return document

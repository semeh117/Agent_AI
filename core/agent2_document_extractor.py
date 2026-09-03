"""Layout-aware CV document extraction used only by Agent 2.

This module deliberately stops at document extraction. It does not call an
LLM, identify skills, normalize ESCO concepts, or calculate compatibility.
It exposes Docling's structured view and PyPDF's sequential view so Agent 2
can use each representation for the fields it handles most reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import html
from pathlib import Path
import re
import unicodedata

from pydantic import BaseModel, Field

from core.extraction_cache import get_cached, set_cached


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

EXTRACTION_VERSION = "agent2-hybrid-extractor-v1"


@dataclass(frozen=True)
class Agent2Document:
    """Two complementary text views exported from one CV document."""

    markdown: str
    plain_text: str
    pypdf_text: str
    backend: str
    source_path: str
    content_hash: str
    extraction_version: str
    detected_sections: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class _CachedAgent2Document(BaseModel):
    """Serializable extraction payload stored independently from its path."""

    markdown: str
    plain_text: str
    pypdf_text: str
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
        markdown=cached.markdown,
        plain_text=cached.plain_text,
        pypdf_text=cached.pypdf_text,
        backend=cached.backend,
        source_path=str(source_path),
        content_hash=cached.content_hash,
        extraction_version=cached.extraction_version,
        detected_sections=tuple(cached.detected_sections),
        warnings=tuple(cached.warnings),
    )


def _document_for_cache(document: Agent2Document) -> _CachedAgent2Document:
    return _CachedAgent2Document(
        markdown=document.markdown,
        plain_text=document.plain_text,
        pypdf_text=document.pypdf_text,
        backend=document.backend,
        content_hash=document.content_hash,
        extraction_version=document.extraction_version,
        detected_sections=list(document.detected_sections),
        warnings=list(document.warnings),
    )


def _normalize_text(value: str) -> str:
    """Normalize Unicode/newlines while preserving Markdown structure."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    return "\n".join(lines).strip()


def clean_docling_text(value: str, *, markdown: bool) -> str:
    """Remove deterministic export artifacts without changing CV content."""

    text = html.unescape(_normalize_text(value))

    # Docling can preserve a PDF line-wrap hyphen as ``famil- iar`` or split
    # one word over two Markdown paragraphs as ``recommenda-\n\ntions``.
    text = re.sub(
        r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])-[ \t]*(?:\n[ \t]*)+(?=[a-zà-öø-ÿ])",
        "",
        text,
    )
    text = re.sub(
        r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])-[ \t]+(?=[a-zà-öø-ÿ])",
        "",
        text,
    )

    if markdown:
        # Contact separators are exported as isolated table bars, and the
        # Projects section label can become a false final row in the skills
        # table. Neither carries candidate information.
        text = re.sub(r"(?m)^\s*\|\s*$\n?", "", text)
        text = re.sub(
            r"(?im)^\s*\|\s*projects\s*\|\s*projects\s*\|\s*$\n?",
            "",
            text,
        )
        text = re.sub(r"\[\|\s*", "[", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_cv_sections(markdown: str) -> tuple[str, ...]:
    """Return recognizable CV headings in their document order."""

    found: list[str] = []
    seen: set[str] = set()

    for raw_line in _normalize_text(markdown).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        candidate = heading_match.group(1) if heading_match else line
        candidate = re.sub(r"^\*\*(.*?)\*\*$", r"\1", candidate).strip(" :-")
        key = candidate.casefold()

        is_markdown_heading = heading_match is not None
        is_known_standalone_heading = key in _KNOWN_CV_SECTIONS
        if not (is_markdown_heading or is_known_standalone_heading):
            continue
        if key in seen:
            continue

        found.append(candidate)
        seen.add(key)

    return tuple(found)


@lru_cache(maxsize=1)
def _get_docling_converter():
    """Create one reusable converter and give a clear dependency error."""

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Docling is not installed. Install the project requirements before "
            "running Agent 2's document-extraction comparison."
        ) from exc

    return DocumentConverter(allowed_formats=[InputFormat.PDF])


def _extract_pypdf_text(source_path: Path) -> str:
    """Extract a stable sequential-text view for metadata validation."""

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyPDF is required for Agent 2's hybrid CV extraction."
        ) from exc

    reader = PdfReader(source_path)
    return _normalize_text(
        "\n".join(page.extract_text() or "" for page in reader.pages)
    )


def extract_cv_document_agent2(
    pdf_source: str | Path,
    *,
    use_cache: bool = True,
) -> Agent2Document:
    """Extract complementary Docling and PyPDF views of one local CV."""

    source_path = Path(pdf_source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CV PDF does not exist: {source_path}")
    if source_path.suffix.casefold() != ".pdf":
        raise ValueError(f"Agent 2 document extraction expects a PDF: {source_path}")

    content_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    cache_identity = f"{EXTRACTION_VERSION}:{content_hash}"
    if use_cache:
        cached = get_cached(
            "agent2_document",
            cache_identity,
            _CachedAgent2Document,
        )
        if (
            cached is not None
            and cached.content_hash == content_hash
            and cached.extraction_version == EXTRACTION_VERSION
        ):
            print(
                "  [CACHE HIT] Agent 2 document already extracted — "
                "Docling and RapidOCR skipped."
            )
            return _document_from_cache(cached, source_path)

    converter = _get_docling_converter()
    result = converter.convert(source_path)
    markdown = clean_docling_text(
        result.document.export_to_markdown(), markdown=True
    )
    plain_text = clean_docling_text(result.document.export_to_text(), markdown=False)
    pypdf_text = _extract_pypdf_text(source_path)

    if not markdown or not plain_text:
        raise ValueError(f"Docling extracted no usable text from: {source_path}")
    if not pypdf_text:
        raise ValueError(f"PyPDF extracted no usable text from: {source_path}")

    sections = detect_cv_sections(markdown)
    warnings: list[str] = []
    if not sections:
        warnings.append("Docling did not expose any recognizable CV section headings.")
    if len(plain_text) < 200:
        warnings.append(
            f"The extracted CV is unusually short ({len(plain_text)} characters)."
        )
    if len(pypdf_text) < 200:
        warnings.append(
            f"PyPDF extracted unusually short text ({len(pypdf_text)} characters)."
        )

    document = Agent2Document(
        markdown=markdown,
        plain_text=plain_text,
        pypdf_text=pypdf_text,
        backend="docling+pypdf",
        source_path=str(source_path),
        content_hash=content_hash,
        extraction_version=EXTRACTION_VERSION,
        detected_sections=sections,
        warnings=tuple(warnings),
    )
    if use_cache:
        set_cached("agent2_document", cache_identity, _document_for_cache(document))
    return document

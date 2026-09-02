"""Capture one real CV input and its parser output for quality review.

Usage:
    python -m dev.capture_cv_parser_fixture cv/AIengineer.pdf
    python -m dev.capture_cv_parser_fixture cv/AIengineer.pdf --use-cache
    python -m dev.capture_cv_parser_fixture --reparse-existing

The fixture deliberately includes the extracted raw CV text so every parsed
field can be checked against its source. Do not commit it if the CV is private.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.agent2_document_extractor import extract_cv_document_agent2
from core.agent2_parser import CV_CACHE_VERSION, extract_cv_info_agent2


DEFAULT_OUTPUT = Path("fixtures/cv_parser_fixture.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture raw CV text and the configured parser's CVInfo."
    )
    parser.add_argument("cv_path", nargs="?", type=Path, help="Source CV PDF path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reparse-existing",
        action="store_true",
        help="Parse raw_text already stored in --out instead of reading a PDF.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached extraction. Omit to test the current parser model fresh.",
    )
    args = parser.parse_args()

    if args.reparse_existing:
        if not args.out.is_file():
            raise FileNotFoundError(f"Existing CV fixture not found: {args.out}")
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        raw_text = str(existing.get("raw_text") or "").strip()
        if not raw_text:
            raise ValueError(f"Existing fixture has no raw_text: {args.out}")
        source_file = str(existing.get("source_file") or "existing fixture")
        document_extractor = str(
            existing.get("document_extractor") or "existing fixture"
        )
        detected_sections = list(existing.get("detected_sections") or [])
        extraction_warnings = list(existing.get("extraction_warnings") or [])
        layout_text = str(existing.get("layout_text") or raw_text).strip()
        content_hash = str(existing.get("content_hash") or "").strip() or None
        extraction_version = str(
            existing.get("extraction_version") or "existing fixture"
        )
    else:
        if args.cv_path is None:
            parser.error("cv_path is required unless --reparse-existing is used.")
        document = extract_cv_document_agent2(args.cv_path)
        raw_text = document.pypdf_text
        layout_text = document.markdown
        content_hash = document.content_hash
        extraction_version = document.extraction_version
        source_file = args.cv_path.name
        document_extractor = document.backend
        detected_sections = list(document.detected_sections)
        extraction_warnings = list(document.warnings)

    parsed = extract_cv_info_agent2(
        raw_text,
        layout_text=layout_text,
        cache_identity=(
            f"{extraction_version}:{content_hash}" if content_hash else None
        ),
        use_cache=args.use_cache,
    )
    fixture = {
        "fixture_type": "cv_parser",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "document_extractor": document_extractor,
        "detected_sections": detected_sections,
        "extraction_warnings": extraction_warnings,
        "parser_provider": os.getenv("PARSER_PROVIDER", "openrouter"),
        "parser_model": os.getenv("PARSER_MODEL", "qwen/qwen-2.5-7b-instruct"),
        "parser_version": CV_CACHE_VERSION,
        "used_extraction_cache": args.use_cache,
        "raw_text": raw_text,
        "layout_text": layout_text,
        "content_hash": content_hash,
        "extraction_version": extraction_version,
        "parsed_cv": parsed.model_dump(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Captured CV parser fixture: {args.out}")
    print(
        f"Parsed {parsed.full_name or 'unknown candidate'}: "
        f"{len(parsed.skills)} skills, {parsed.experience_years} years, "
        f"education={parsed.highest_education_level or 'not found'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

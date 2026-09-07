"""Interactive CV parsing review for the Agent 2 hybrid pipeline.

Lists the PDFs available in the ``cv/`` folder, lets the user pick one, and
prints the document-extraction and structured-parsing results side by side.

Usage:
    python dev/inspect_cv_parsing.py
    python dev/inspect_cv_parsing.py --no-cache
    python dev/inspect_cv_parsing.py --cv cv/Semah_Mechi_.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent2_document_extractor import extract_cv_document_agent2  # noqa: E402
from core.agent2_parser import extract_cv_info_agent2  # noqa: E402


CV_DIR = PROJECT_ROOT / "cv"


def _list_cvs() -> list[Path]:
    return sorted(
        path for path in CV_DIR.glob("*.pdf") if path.is_file()
    )


def _choose_cv(cvs: list[Path]) -> Path:
    print("Available CVs in the 'cv' folder:")
    for index, path in enumerate(cvs, start=1):
        print(f"  [{index}] {path.name}")
    while True:
        raw = input("\nPick a CV number: ").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= len(cvs):
            print(f"Please choose a number between 1 and {len(cvs)}.")
            continue
        return cvs[int(raw) - 1]


def _print_extraction(cv_path: Path, use_cache: bool) -> None:
    print("\n" + "=" * 72)
    print("STAGE 1 - DOCUMENT EXTRACTION (Docling + PyPDF, no LLM)")
    print("=" * 72)
    document = extract_cv_document_agent2(cv_path, use_cache=use_cache)

    print(f"Backend: {document.backend}")
    print(f"Content SHA-256: {document.content_hash[:16]}...")
    print(f"Extraction version: {document.extraction_version}")
    print(f"PyPDF characters: {len(document.pypdf_text)}")
    print(f"Docling Markdown characters: {len(document.markdown)}")

    print("\nDetected sections:")
    if document.detected_sections:
        for section in document.detected_sections:
            print(f"  - {section}")
    else:
        print("  - none detected")

    if document.warnings:
        print("\nWarnings:")
        for warning in document.warnings:
            print(f"  ! {warning}")


def _print_evidence(skills: list[str], evidence: dict[str, str]) -> None:
    for skill in skills:
        excerpt = evidence.get(skill, "")
        if excerpt:
            print(f"  - {skill!r}  (evidence: ...{excerpt[:90]}...)")
        else:
            print(f"  - {skill!r}  (no source excerpt)")


def _print_parse(cv_path: Path, use_cache: bool) -> None:
    print("\n" + "=" * 72)
    print("STAGE 2 - STRUCTURED PARSE (LLM + deterministic grounding)")
    print("=" * 72)
    document = extract_cv_document_agent2(cv_path, use_cache=True)
    cv_info = extract_cv_info_agent2(
        document.pypdf_text,
        layout_text=document.markdown,
        cache_identity=(
            f"{document.extraction_version}:{document.content_hash}"
        ),
        use_cache=use_cache,
    )

    print(
        f"\nName: {cv_info.full_name or 'not found'}"
        f"  |  Headline: {cv_info.headline or 'not found'}"
    )
    print(f"Job titles: {', '.join(cv_info.job_titles) or 'none'}")
    print(f"Experience: {cv_info.experience_years} years")
    print(
        f"Highest education: "
        f"{cv_info.highest_education_level or 'not found'}"
    )
    print(f"Education entries: {len(cv_info.education)}")
    for entry in cv_info.education:
        print(f"  - {entry}")
    print(f"Mail: {cv_info.mail or 'not found'}")
    print(f"Phone: {cv_info.phone or 'not found'}")
    print(f"LinkedIn: {cv_info.linkedin or 'not found'}")
    print(f"GitHub: {cv_info.github or 'not found'}")

    print(f"\nSkills ({len(cv_info.skills)}):")
    _print_evidence(cv_info.skills, cv_info.skill_evidence)
    if not cv_info.skills:
        print("  - none")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively review Agent 2 CV parsing on a pick."
    )
    parser.add_argument(
        "--cv",
        type=Path,
        default=None,
        help="Path to a CV PDF; omit to choose from the cv folder.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh extraction and a fresh parser LLM call.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    use_cache = not args.no_cache

    if args.cv is not None:
        cv_path = args.cv
        if not cv_path.is_file():
            raise FileNotFoundError(f"CV PDF does not exist: {cv_path}")
    else:
        cvs = _list_cvs()
        if not cvs:
            raise FileNotFoundError(f"No PDFs found in {CV_DIR}.")
        cv_path = _choose_cv(cvs)

    print(f"CV: {cv_path.resolve()}")
    _print_extraction(cv_path, use_cache=use_cache)
    _print_parse(cv_path, use_cache=use_cache)
    print("\nDone. Remove --no-cache on reruns to reuse cached parsing.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
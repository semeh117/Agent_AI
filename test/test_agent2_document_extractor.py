"""Manual PyPDF-versus-Docling extraction comparison for Agent 2.

This test makes no LLM, ESCO, embedding, LinkedIn, Gmail, or Telegram calls.
It exists to answer one question before integration: does Docling preserve the
CV's layout and content better than the current flat PyPDF extraction?

Run:
    python test/test_agent2_document_extractor.py --cv cv/Semah_Mechi_.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent2_document_extractor import (  # noqa: E402
    detect_cv_sections,
    extract_cv_document_agent2,
)
from core.cv_parser import extract_text_from_pdf  # noqa: E402
import core.agent2_document_extractor as document_extractor  # noqa: E402
import core.extraction_cache as extraction_cache  # noqa: E402


DEFAULT_CV = PROJECT_ROOT / "cv" / "Semah_Mechi_.pdf"
EXPECTED_TERMS = (
    "Python",
    "Scikit-learn",
    "PyTorch",
    "XGBoost",
    "LangChain",
    "ChromaDB",
    "FastAPI",
    "PostgreSQL",
    "MongoDB",
    "Docker",
)
KNOWN_JOINED_ARTIFACTS = (
    "MLPython",
    "DatabasesPostgreSQL",
    "RAGLangChain",
    "DeployFastAPI",
)


@dataclass(frozen=True)
class ExtractionMetrics:
    characters: int
    nonempty_lines: int
    section_headings: int
    expected_terms_found: int
    joined_artifacts: int
    distinct_years: int


@dataclass(frozen=True)
class ParserMetrics:
    skills_count: int
    expected_terms_found: int
    missing_expected_terms: tuple[str, ...]
    suspicious_skills: tuple[str, ...]
    experience_years: float | None


def _contains(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def _metrics(text: str) -> ExtractionMetrics:
    lines = [line for line in text.splitlines() if line.strip()]
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return ExtractionMetrics(
        characters=len(text),
        nonempty_lines=len(lines),
        section_headings=len(detect_cv_sections(text)),
        expected_terms_found=sum(_contains(text, term) for term in EXPECTED_TERMS),
        joined_artifacts=sum(
            _contains(text, artifact) for artifact in KNOWN_JOINED_ARTIFACTS
        ),
        distinct_years=len(years),
    )


def _canonical_skill(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:\+\+|#)?", value.casefold()))


def _skill_is_present(skills: list[str], expected: str) -> bool:
    expected_key = _canonical_skill(expected)
    return any(_canonical_skill(skill) == expected_key for skill in skills)


def _looks_suspicious_skill(skill: str) -> bool:
    text = str(skill or "").strip()
    lowered = text.casefold()
    return bool(
        re.search(r"\b\d+(?:\.\d+)?%", text)
        or lowered.startswith(("and ", "or "))
        or lowered.endswith((" deployed", " per", " late"))
        or "). deployed" in lowered
        or ". audio" in lowered
    )


def _parser_metrics(parsed) -> ParserMetrics:
    skills = list(parsed.skills)
    missing = tuple(
        term for term in EXPECTED_TERMS if not _skill_is_present(skills, term)
    )
    suspicious = tuple(skill for skill in skills if _looks_suspicious_skill(skill))
    return ParserMetrics(
        skills_count=len(skills),
        expected_terms_found=len(EXPECTED_TERMS) - len(missing),
        missing_expected_terms=missing,
        suspicious_skills=suspicious,
        experience_years=parsed.experience_years,
    )


def _print_metric(label: str, pypdf_value: int, docling_value: int) -> None:
    print(f"{label:<27} {pypdf_value:>10} {docling_value:>10}")


def _print_term_review(pypdf_text: str, docling_text: str) -> None:
    print("\nEXPECTED TERM COVERAGE")
    print(f"{'term':<24} {'PyPDF':>8} {'Docling':>9}")
    for term in EXPECTED_TERMS:
        print(
            f"{term:<24} "
            f"{('yes' if _contains(pypdf_text, term) else 'no'):>8} "
            f"{('yes' if _contains(docling_text, term) else 'no'):>9}"
        )


def _print_artifact_review(pypdf_text: str, docling_text: str) -> None:
    print("\nKNOWN JOINED-TEXT ARTIFACTS")
    print(f"{'artifact':<24} {'PyPDF':>8} {'Docling':>9}")
    for artifact in KNOWN_JOINED_ARTIFACTS:
        print(
            f"{artifact:<24} "
            f"{('found' if _contains(pypdf_text, artifact) else 'clear'):>8} "
            f"{('found' if _contains(docling_text, artifact) else 'clear'):>9}"
        )


def compare_extractors(cv_path: Path, check_determinism: bool = True) -> bool:
    """Print the comparison and return whether Docling was deterministic."""

    print("AGENT 2 CV DOCUMENT EXTRACTION COMPARISON")
    print(f"CV: {cv_path.resolve()}")
    print("No LLM, ESCO, embedding, or agent calls are made.\n")

    pypdf_text = extract_text_from_pdf(str(cv_path))
    docling_result = extract_cv_document_agent2(cv_path)
    docling_text = docling_result.markdown

    pypdf_metrics = _metrics(pypdf_text)
    docling_metrics = _metrics(docling_text)

    print(f"{'metric':<27} {'PyPDF':>10} {'Docling':>10}")
    _print_metric("Characters", pypdf_metrics.characters, docling_metrics.characters)
    _print_metric(
        "Non-empty lines", pypdf_metrics.nonempty_lines, docling_metrics.nonempty_lines
    )
    _print_metric(
        "Recognizable headings",
        pypdf_metrics.section_headings,
        docling_metrics.section_headings,
    )
    _print_metric(
        "Expected terms found",
        pypdf_metrics.expected_terms_found,
        docling_metrics.expected_terms_found,
    )
    _print_metric(
        "Joined artifacts",
        pypdf_metrics.joined_artifacts,
        docling_metrics.joined_artifacts,
    )
    _print_metric(
        "Distinct years", pypdf_metrics.distinct_years, docling_metrics.distinct_years
    )

    _print_term_review(pypdf_text, docling_text)
    _print_artifact_review(pypdf_text, docling_text)

    print("\nDOCLING SECTIONS")
    if docling_result.detected_sections:
        for section in docling_result.detected_sections:
            print(f"  - {section}")
    else:
        print("  - None detected")

    if docling_result.warnings:
        print("\nDOCLING WARNINGS")
        for warning in docling_result.warnings:
            print(f"  - {warning}")

    deterministic = True
    if check_determinism:
        second = extract_cv_document_agent2(cv_path)
        deterministic = (
            second.markdown == docling_result.markdown
            and second.plain_text == docling_result.plain_text
        )
        print(f"\nDeterministic repeated extraction: {'PASS' if deterministic else 'FAIL'}")

    no_worse_on_terms = (
        docling_metrics.expected_terms_found >= pypdf_metrics.expected_terms_found
    )
    no_worse_on_artifacts = (
        docling_metrics.joined_artifacts <= pypdf_metrics.joined_artifacts
    )
    better_structure = (
        docling_metrics.section_headings > pypdf_metrics.section_headings
        or docling_metrics.nonempty_lines > pypdf_metrics.nonempty_lines
    )

    print("\nPRELIMINARY VERDICT")
    if deterministic and no_worse_on_terms and no_worse_on_artifacts and better_structure:
        print("Docling is a promising Agent 2 input and should proceed to parser testing.")
    else:
        print("Docling has not yet demonstrated a clear improvement; review the output.")

    return deterministic


def compare_agent2_parsers(cv_path: Path) -> None:
    """Parse both extraction formats with the same LLM and no cache."""

    from core.agent2_parser import extract_cv_info_agent2

    print("\n" + "=" * 80)
    print("AGENT 2 PARSER A/B COMPARISON")
    print("Same CV, same parser model, cache disabled")
    print("=" * 80)

    pypdf_text = extract_text_from_pdf(str(cv_path))
    docling_markdown = extract_cv_document_agent2(cv_path).markdown

    print("\n[A] Parsing PyPDF text...")
    pypdf_parsed = extract_cv_info_agent2(pypdf_text, use_cache=False)
    print("[B] Parsing cleaned Docling Markdown...")
    docling_parsed = extract_cv_info_agent2(docling_markdown, use_cache=False)

    pypdf_review = _parser_metrics(pypdf_parsed)
    docling_review = _parser_metrics(docling_parsed)

    print("\nPARSER METRICS")
    print(f"{'metric':<30} {'PyPDF':>12} {'Docling':>12}")
    print(
        f"{'Skills returned':<30} "
        f"{pypdf_review.skills_count:>12} {docling_review.skills_count:>12}"
    )
    print(
        f"{'Expected terms found':<30} "
        f"{pypdf_review.expected_terms_found:>12} "
        f"{docling_review.expected_terms_found:>12}"
    )
    print(
        f"{'Suspicious skills':<30} "
        f"{len(pypdf_review.suspicious_skills):>12} "
        f"{len(docling_review.suspicious_skills):>12}"
    )
    print(
        f"{'Experience years':<30} "
        f"{str(pypdf_review.experience_years):>12} "
        f"{str(docling_review.experience_years):>12}"
    )

    for label, parsed, review in (
        ("PyPDF", pypdf_parsed, pypdf_review),
        ("Docling", docling_parsed, docling_review),
    ):
        print(f"\n{label.upper()} PARSED SKILLS ({review.skills_count})")
        for skill in parsed.skills:
            print(f"  - {skill}")
        print(
            "  Missing expected: "
            + (", ".join(review.missing_expected_terms) or "None")
        )
        print(
            "  Suspicious: "
            + (", ".join(review.suspicious_skills) or "None")
        )

    docling_wins = (
        docling_review.expected_terms_found >= pypdf_review.expected_terms_found
        and len(docling_review.suspicious_skills)
        <= len(pypdf_review.suspicious_skills)
        and (
            docling_review.expected_terms_found > pypdf_review.expected_terms_found
            or len(docling_review.suspicious_skills)
            < len(pypdf_review.suspicious_skills)
        )
        and (
            docling_review.experience_years is not None
            and 0.1 <= docling_review.experience_years <= 0.25
        )
    )
    print("\nPARSER A/B VERDICT")
    if docling_wins:
        print("Docling wins and is ready to become Agent 2's PDF input.")
    else:
        print("Docling has not yet met every adoption criterion.")


def test_agent2_document_extraction_cache() -> None:
    """Verify content/version invalidation without loading Docling or OCR."""

    class _FakeDocument:
        @staticmethod
        def export_to_markdown() -> str:
            return "# Skills\nPython\n" + ("Structured CV content. " * 15)

        @staticmethod
        def export_to_text() -> str:
            return "Skills\nPython\n" + ("Sequential CV content. " * 15)

    class _FakeResult:
        document = _FakeDocument()

    class _FakeConverter:
        def __init__(self) -> None:
            self.calls = 0

        def convert(self, _source_path: Path) -> _FakeResult:
            self.calls += 1
            return _FakeResult()

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        pdf_path = temporary_path / "candidate.pdf"
        cache_path = temporary_path / "cache"
        pdf_path.write_bytes(b"%PDF-1.4 same candidate content")
        converter = _FakeConverter()

        with (
            patch.object(extraction_cache, "CACHE_DIR", cache_path),
            patch.object(
                document_extractor,
                "_get_docling_converter",
                return_value=converter,
            ),
            patch.object(
                document_extractor,
                "_extract_pypdf_text",
                return_value="PyPDF candidate content. " * 15,
            ),
        ):
            first = document_extractor.extract_cv_document_agent2(pdf_path)
            second = document_extractor.extract_cv_document_agent2(pdf_path)
            assert converter.calls == 1
            assert second.markdown == first.markdown
            assert second.content_hash == first.content_hash

            pdf_path.write_bytes(b"%PDF-1.4 changed candidate content")
            changed = document_extractor.extract_cv_document_agent2(pdf_path)
            assert converter.calls == 2
            assert changed.content_hash != first.content_hash

            with patch.object(
                document_extractor,
                "EXTRACTION_VERSION",
                "agent2-hybrid-extractor-cache-test-v2",
            ):
                version_changed = document_extractor.extract_cv_document_agent2(
                    pdf_path
                )
            assert converter.calls == 3
            assert version_changed.extraction_version.endswith("v2")

            document_extractor.extract_cv_document_agent2(
                pdf_path,
                use_cache=False,
            )
            assert converter.calls == 4

    print("Agent 2 document extraction cache: PASS")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyPDF and Docling extraction on one CV PDF."
    )
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV)
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip the second Docling conversion when doing a quick review.",
    )
    parser.add_argument(
        "--compare-parser",
        action="store_true",
        help=(
            "Also make two live parser-LLM calls (PyPDF and Docling) with "
            "the extraction cache disabled."
        ),
    )
    parser.add_argument(
        "--test-cache",
        action="store_true",
        help="Test extraction-cache hits and invalidation without Docling/OCR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.test_cache:
        test_agent2_document_extraction_cache()
        return 0
    if not args.cv.is_file():
        raise FileNotFoundError(f"CV PDF does not exist: {args.cv}")
    deterministic = compare_extractors(
        args.cv, check_determinism=not args.skip_determinism
    )
    if args.compare_parser:
        compare_agent2_parsers(args.cv)
    return 0 if deterministic else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

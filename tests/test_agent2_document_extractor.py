"""Offline cache regression; live comparison lives in examples/."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import next_chapter.parsing.agent2_document_extractor as document_extractor
import next_chapter.storage.extraction_cache as extraction_cache


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

if __name__ == "__main__":
    test_agent2_document_extraction_cache()

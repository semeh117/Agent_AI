"""Offline regression tests for lightweight CV document extraction."""

from pathlib import Path
import tempfile
from unittest.mock import patch

import next_chapter.parsing.agent2_document_extractor as document_extractor
import next_chapter.storage.extraction_cache as extraction_cache
from reportlab.pdfgen import canvas


def test_agent2_document_extraction_cache() -> None:
    """Verify content/version invalidation without reading a real PDF."""

    extracted_text = "Skills\nPython\n" + ("Candidate experience. " * 15)
    calls = 0

    def fake_extract(_source_path: Path) -> str:
        nonlocal calls
        calls += 1
        return extracted_text

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        pdf_path = temporary_path / "candidate.pdf"
        cache_path = temporary_path / "cache"
        pdf_path.write_bytes(b"%PDF-1.4 same candidate content")

        with (
            patch.object(extraction_cache, "CACHE_DIR", cache_path),
            patch.object(
                document_extractor,
                "_extract_pypdf_text",
                side_effect=fake_extract,
            ),
        ):
            first = document_extractor.extract_cv_document_agent2(pdf_path)
            second = document_extractor.extract_cv_document_agent2(pdf_path)
            assert calls == 1
            assert second.text == first.text
            assert second.backend == "pypdf"
            assert second.content_hash == first.content_hash

            pdf_path.write_bytes(b"%PDF-1.4 changed candidate content")
            changed = document_extractor.extract_cv_document_agent2(pdf_path)
            assert calls == 2
            assert changed.content_hash != first.content_hash

            with patch.object(
                document_extractor,
                "EXTRACTION_VERSION",
                "agent2-pypdf-extractor-cache-test-v3",
            ):
                version_changed = document_extractor.extract_cv_document_agent2(
                    pdf_path
                )
            assert calls == 3
            assert version_changed.extraction_version.endswith("v3")

            document_extractor.extract_cv_document_agent2(
                pdf_path,
                use_cache=False,
            )
            assert calls == 4

    print("Agent 2 document extraction cache: PASS")


def test_scanned_pdf_error_is_clear() -> None:
    """Image-only PDFs should explain how the user can make them usable."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        pdf_path = Path(temporary_directory) / "scanned.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 image-only placeholder")
        with patch.object(document_extractor, "_extract_pypdf_text", return_value=""):
            try:
                document_extractor.extract_cv_document_agent2(
                    pdf_path,
                    use_cache=False,
                )
            except ValueError as exc:
                message = str(exc)
                assert "no selectable text" in message
                assert "scanned image" in message
            else:
                raise AssertionError("Image-only PDF should have been rejected.")


def test_text_pdf_is_extracted_with_pypdf() -> None:
    """Exercise the real lightweight backend on a generated text PDF."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        pdf_path = Path(temporary_directory) / "text-cv.pdf"
        document_canvas = canvas.Canvas(str(pdf_path))
        document_canvas.drawString(72, 760, "Skills")
        document_canvas.drawString(72, 740, "Python and SQL")
        document_canvas.save()

        document = document_extractor.extract_cv_document_agent2(
            pdf_path,
            use_cache=False,
        )
        assert document.backend == "pypdf"
        assert "Skills" in document.text
        assert "Python and SQL" in document.text


if __name__ == "__main__":
    test_agent2_document_extraction_cache()
    test_scanned_pdf_error_is_clear()
    test_text_pdf_is_extracted_with_pypdf()

# test/test_send_results_email_docx.py
"""
test_send_results_email_docx.py
----------------------
Verifies the cover letter is rendered into a real, re-openable .docx
attachment payload (python-docx) with a safe filename — the "send the cover
letter in .docs/.docx format" requirement.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from pipeline.send_results_email import (
    _build_cover_letter_docx,
    _safe_attachment_name,
)


class FakeCv:
    full_name = "Jane Doe"


LETTER = (
    "Dear Hiring Team,\n\n"
    "I am excited to apply for the Senior AI Engineer position at Acme.\n\n"
    "Best regards,\nJane Doe"
)


def test_docx_bytes_are_a_valid_docx():
    data = _build_cover_letter_docx(FakeCv(), LETTER, "Senior AI Engineer", "Acme")
    assert data[:2] == b"PK"  # .docx is a zip container

    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Cover Letter" in text
    assert "Jane Doe" in text
    assert "Senior AI Engineer" in text
    assert "Acme" in text
    assert "I am excited to apply" in text


def test_safe_attachment_name():
    name = _safe_attachment_name("AI Engineer", "Acme")
    assert name == "Cover Letter - AI Engineer @ Acme.docx"
    assert name.endswith(".docx")
    # illegal filename characters are stripped
    assert ":" not in _safe_attachment_name("AI/ML:Engineer", "Product? Co|")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All docx tests passed.")

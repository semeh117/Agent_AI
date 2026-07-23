"""
cv_parser.py
--------------
CV Parsing & Extraction (Feature #1) — PDF only.

Stage 1: extract_text_from_pdf() — deterministic, pypdf, no LLM.
Stage 2: extract_cv_info() — LLM-powered, forced structured output.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from pypdf import PdfReader
from config import get_llm


def extract_text_from_pdf(pdf_source) -> str:
    """
    pdf_source: file path (str) or file-like object (Streamlit UploadedFile).
    pypdf's PdfReader accepts both.
    """
    reader = PdfReader(pdf_source)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.strip()

    if not text:
        raise ValueError(
            "No extractable text found. Likely a scanned/image-based PDF "
            "(needs OCR, out of scope for this prototype)."
        )
    return text


class CVInfo(BaseModel):
    full_name: Optional[str] = Field(default=None, description="Candidate's full name, else null.")
    skills: List[str] = Field(
        description="Concrete technical/professional skills mentioned (tools, "
                    "languages, frameworks, methodologies). Short entries, e.g. 'Python'."
    )
    job_titles: List[str] = Field(description="Past job titles/roles/internships held.")
    experience_years: Optional[int] = Field(
        default=None, description="Estimated total years of professional experience. Null if unclear."
    )
    education: List[str] = Field(default_factory=list, description="Degrees/certifications mentioned.")
    linkedin : Optional[str] = Field(default=None, description="LinkedIn profile URL if found, else null.")
    mail : Optional[str] = Field(default=None, description="Email address if found, else null.")
    github : Optional[str] = Field(default=None, description="GitHub profile URL if found, else null.")


def extract_cv_info(cv_text: str, llm=None) -> CVInfo:
    if llm is None:
        llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(CVInfo)

    prompt = f"""You are parsing a CV. You MUST read the ENTIRE document below,
including every section: Professional Summary, Experience, Projects,
Certifications, Technical Skills, and Education. Do not stop after the
first section.

Extract:
- full_name: the person's name, usually on the very first line of the CV.
- skills: EVERY technical skill, tool, framework, programming language, and
  library mentioned ANYWHERE in the document — including inside project
  descriptions (e.g. "PyTorch", "XGBoost") AND the explicit Technical Skills
  section (e.g. "Python", "SQL", "Flutter"). Do not limit yourself to the
  Professional Summary paragraph. Aim to capture 15-25 distinct skills for
  a technical CV like this one.
- job_titles: actual roles/positions/internships held (found under
  "Experience"), NOT a professional tagline or headline.
- experience_years: estimate from Experience section dates only; null if
  the candidate has no significant professional history yet (e.g. student).
- education: degrees, specializations, and certifications.

Note:
-raw text may have minor extraction artifacts (missing spaces between
merged sections, misplaced accent characters like ´E instead of É) — read
through these naturally, they don't change the meaning.
-Only extract skills that are explicitly written in the text. Do not infer related or implied skills that aren't literally mentioned
CV TEXT:
---
{cv_text}
---"""
    result = structured_llm.invoke(prompt)
    seen = set()
    deduped = []
    for s in result.skills:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s.strip())
    result.skills = deduped
    return result
def verify_skills_grounded(skills: list[str], raw_text: str) -> list[str]:
    """
    Keeps only skills that actually appear (case-insensitive substring match)
    in the raw CV text — filters out LLM hallucinations that weren't
    actually in the source document.
    """
    text_lower = raw_text.lower()
    return [s for s in skills if s.lower() in text_lower]

    
    



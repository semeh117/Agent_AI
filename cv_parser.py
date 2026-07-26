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
    experience_years: Optional[float] = Field(
        default=None,
        description="Total years of professional experience, summed from actual "
                    "dated Experience entries (include internships/freelance — do "
                    "NOT exclude by title wording, use real durations). 0 if none."
    )
    education: List[str] = Field(default_factory=list, description="Degrees/certifications mentioned.")
    highest_education_level: Optional[str] = Field(
        default=None,
        description="The candidate's highest completed OR in-progress education "
                    "level, normalized to exactly one of: 'High School', "
                    "'Bachelor', 'Master', 'PhD'. For an ongoing multi-year "
                    "engineering/university program, infer the level it leads to "
                    "(e.g. a 5-year integrated engineering degree -> 'Master'). "
                    "Null only if genuinely no education is mentioned."
    )
    phone: Optional[str] = Field(default=None, description="Phone number if found, else null.")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL if found, else null.")
    mail: Optional[str] = Field(default=None, description="Email address if found, else null.")
    github: Optional[str] = Field(default=None, description="GitHub profile URL if found, else null.")


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
- experience_years: sum the actual duration of EVERY entry in the
  Experience section from its stated dates (use today's date for entries
  marked "Present"). Include internships and freelance work — do NOT
  exclude time just because a role is titled "Internship". Return 0 only
  if there is no Experience section at all.
- education: degrees, specializations, and certifications, as listed.
- highest_education_level: normalize the candidate's highest level to
  exactly one of 'High School', 'Bachelor', 'Master', 'PhD' — even if the
  CV describes it differently (e.g. a 5-year "Integrated Engineering
  Cycle" culminating in an engineering degree should map to 'Master',
  since it's equivalent to a Bachelor's + Master's combined).
-linkedin: LinkedIn profile URL if found, else null.
-mail: Email address if found, else null.
-phone: Phone number if found, else null.
-github: GitHub profile URL if found, else null.

Note:
-raw text may have minor extraction artifacts (missing spaces between
merged sections, misplaced accent characters like ´E instead of É) — read
through these naturally, they don't change the meaning.
-Only extract skills that are explicitly written in the text. Do not infer related or implied skills that aren't literally mentioned
CV TEXT:
---
{cv_text}
---"""
    try:
        result = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"  [WARN] CV extraction failed (likely token limit): {str(e)[:150]}")
        # Return an empty/minimal CVInfo rather than crashing the whole pipeline
        return CVInfo(full_name=None, skills=[], job_titles=[],
                       experience_years=None, education=[],
                       highest_education_level=None)

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

    
    



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
from config import get_parser_llm
from core.extraction_cache import get_cached, set_cached

# Default cache behavior. Can be overridden by the application/config layer.



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


def extract_cv_info(cv_text: str, llm=None ,use_cache :bool= True) -> CVInfo:
    if use_cache:
        cached = get_cached("cv", cv_text, CVInfo)
        if cached is not None:
            print(f"  [CACHE HIT] CV already parsed before — skipping LLM call "
                  f"({len(cached.skills)} skills, no wait).")
            return cached
    if llm is None:
        llm = get_parser_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(CVInfo)

    prompt = f"""You are an expert CV/resume parser. Extract structured information
from the ENTIRE CV below, not just the first section.

Read the entire document from beginning to end before producing your
answer. You must inspect every section that appears, including but not
limited to: Professional Summary, Experience, Projects, Certifications,
Technical Skills, Education, and any other sections. Do not stop after
the first section, summary, or skills list.

## full_name
The candidate's full name, normally near the beginning of the CV. Do
NOT use a job title, professional headline/tagline, company name, or
email address. Return null if a name cannot be identified confidently.

## skills
Extract EVERY distinct technical skill explicitly mentioned anywhere in
the CV — Professional Summary, Experience descriptions, Project
descriptions, Technical Skills section, Certifications, Education, and
any other section. Include programming languages, frameworks,
libraries, databases, cloud platforms, developer tools, technical
platforms, technical methodologies, ML/AI technologies, and other
explicitly named technical technologies.

For example, if a project description mentions "PyTorch" or "XGBoost",
extract them even if absent from the Technical Skills section. If the
Technical Skills section lists "Python", "SQL", "Flutter", extract
those too.

Do NOT infer skills — only extract technologies explicitly named in the
CV. Normalize obvious capitalization variants so the same skill is
returned once ("python" -> "Python"). Do NOT merge distinct
technologies merely because they're related (Python and Django stay
separate; SQL and PostgreSQL stay separate; React and React Native stay
separate).

Do NOT treat every word in a job title as a skill. "Machine Learning
Engineer" as a job title does NOT automatically mean "Machine Learning"
belongs in skills — only include it if explicitly mentioned elsewhere as
a technology/technical competency.

Do NOT add technologies merely typical for a role. If the CV says
"Developed web applications with React", do not automatically add
JavaScript, HTML, CSS, or Node.js unless explicitly mentioned.

## job_titles
Actual positions held — full-time, part-time, internships, freelance,
contract roles — primarily from the Experience section. Do NOT extract
professional headlines, taglines, desired job titles, skills, or
company names. Preserve the title as written where practical.

## experience_years
Calculate total duration from EVERY Experience entry using dates
explicitly stated. Include internships and freelance work when listed
as Experience entries. For an entry ending in "Present", use
2026-08-14 as the current date. Do not exclude an entry because it's
titled "Intern", "Internship", "Trainee", or similar. Do not count
education, projects, certifications, or unrelated activities as work
experience. Avoid double-counting overlapping employment periods.
Return the result as a decimal (e.g. "January 2022 - January 2024" ->
approximately 2.0). If exact months are available, calculate from the
stated month/year ranges rather than rounding each role independently.
Return 0 only when there is no Experience section or no identifiable
Experience entries.

## education
The candidate's formal education and certifications as listed —
degree, field/specialization, institution, graduation/completion date
when available, and certifications. Do not invent missing information.

## highest_education_level
Normalize to exactly one of: 'High School', 'Bachelor', 'Master', 'PhD'.
A 5-year integrated engineering program explicitly equivalent to a
combined Bachelor's + Master's -> 'Master'. Use the highest EXPLICITLY
supported level — do not infer a higher level solely from job seniority
or years of experience. Return null if education information is
insufficient to determine a level.

## Contact information
linkedin, mail, phone, github — extract only if explicitly present.
Return null for anything not found. Do not fabricate, reconstruct, or
infer contact information. Preserve the actual profile URL found in
the CV for linkedin/github.

## Final validation, before answering
Confirm you have: read the entire CV; checked every section for
technical skills; included technologies from inside project/experience
descriptions, not just the dedicated skills section; deduplicated
skills; extracted actual held job titles rather than headlines;
included internships and freelance entries; calculated experience using
stated dates and 2026-08-14 for "Present"; avoided double-counting
overlapping employment periods; correctly normalized highest education
level; returned null for missing contact info rather than guessing;
avoided inferred or merely implied technical skills.

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
    result.skills = verify_skills_grounded(result.skills, cv_text)

    if use_cache :
        set_cached("cv", cv_text, result)
    return result


def verify_skills_grounded(skills: list[str], raw_text: str) -> list[str]:
    """
    Keeps only skills that actually appear (case-insensitive substring match)
    in the raw CV text — filters out LLM hallucinations that weren't
    actually in the source document.
    """
    text_lower = raw_text.lower()
    return [s for s in skills if s.lower() in text_lower]

    
    



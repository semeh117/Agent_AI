
"""
cover_letter.py
--------------
Generates a cover letter for a candidate's single best-matched job.

Takes the TOP item from job_matching_pipeline.py's run_job_matching()
output directly — that list is already sorted descending by score, so
"the best match" is always just results[0]. No parsing, no digging
through agent reasoning traces; see the project's dev notes on why the
deterministic pipeline's output was chosen over parsing the ReAct
agent's intermediate_steps for this.

This is a plain, standalone LLM call — not part of any agent tool loop —
called once, after ranking is already finished, targeting only the #1
job.
"""

from config import get_llm


COVER_LETTER_PROMPT = """Write a professional, concise cover letter for this candidate
applying to this specific job. Ground every claim in the candidate's
ACTUAL profile below — do not invent experience, skills, or achievements
that aren't listed.

CANDIDATE PROFILE:
- Name: {full_name}
- Most recent title: {job_title}
- Key skills: {skills}
- Experience: {experience_years} years
- Education: {education}

JOB:
- Title: {target_job_title}
- Company: {company}
- Description: {job_description}

MATCH CONTEXT (use this to decide what to emphasize and how to honestly
address any gap — do not hide a real gap, address it constructively):
- Match score: {score_percent}%
- Matching skills: {matching_skills}
- Missing/gap skills: {missing_skills}

Rules:
- 3-4 short paragraphs. No placeholder brackets like [Your Name] — use
  the actual name given above, or omit the line if full_name is missing.
- Open with genuine interest in the specific role/company, not a generic
  greeting.
- Highlight 2-3 of the candidate's ACTUAL matching skills/experience most
  relevant to this job — don't just list everything.
- If there's a meaningful missing skill, address it briefly and honestly
  (e.g. framed as eagerness to learn) rather than ignoring it or pretending
  the candidate already has it.
- Professional tone, no over-the-top enthusiasm or clichés.
- Do not fabricate any detail not present in the candidate profile above."""


def generate_cover_letter(cv_info, top_result: dict, llm=None) -> str:
    """
    Args:
        cv_info: the candidate's CVInfo object (already extracted).
        top_result: results[0] from run_job_matching() — the highest
            scoring job, already shaped with job_title, company,
            description, score_percent, and skills_detail.
        llm: optional pre-built LLM instance; defaults to config.py's
            configured model.

    Returns:
        The cover letter as plain text.
    """
    if llm is None:
        llm = get_llm(temperature=0.3)

    matching_skills = [m["job_skill"] for m in top_result["skills_detail"]["matching"]]
    missing_skills = top_result["skills_detail"]["missing"]

    prompt = COVER_LETTER_PROMPT.format(
        full_name=cv_info.full_name or "the candidate",
        job_title=cv_info.job_titles[0] if cv_info.job_titles else "N/A",
        skills=", ".join(cv_info.skills[:15]),
        experience_years=cv_info.experience_years,
        education=cv_info.highest_education_level or "Not specified",
        target_job_title=top_result["job_title"],
        company=top_result["company"],
        job_description=top_result["description"],
        score_percent=top_result["score_percent"],
        matching_skills=", ".join(matching_skills) or "-",
        missing_skills=", ".join(missing_skills) or "-",
    )

    response = llm.invoke(prompt)
    return response.content
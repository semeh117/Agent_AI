"""Generate, persist, and render Agent 2 interview-preparation packs."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    CondPageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from next_chapter.config import get_interview_llm
from next_chapter.services.application_tracker import get_application, get_candidate_profile
from next_chapter.storage.agent2_database import agent2_connection, initialize_agent2_database


from next_chapter.paths import PROJECT_ROOT
DEFAULT_INTERVIEW_PDF_DIRECTORY = PROJECT_ROOT / "output" / "pdf"


class InterviewQuestion(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    competency: str = Field(max_length=300)
    why_asked: str = Field(min_length=5, max_length=700)
    job_connection: str = Field(max_length=700)
    answer_strategy: str = Field(min_length=5, max_length=1200)
    sample_answer: str = Field(min_length=10, max_length=2200)
    cv_evidence: list[str] = Field(min_length=1, max_length=5)
    learning_plan: str = Field(max_length=700)


class InterviewPreparationContent(BaseModel):
    role_summary: str = Field(min_length=20, max_length=1500)
    role_family: str = Field(max_length=200)
    primary_focus: str = Field(max_length=500)
    key_competencies: list[str] = Field(max_length=6)
    candidate_strengths: list[str] = Field(max_length=6)
    primary_gaps: list[str] = Field(max_length=6)
    technical_questions: list[InterviewQuestion] = Field(min_length=5, max_length=5)
    gap_questions: list[InterviewQuestion] = Field(min_length=2, max_length=2)
    behavioral_questions: list[InterviewQuestion] = Field(min_length=3, max_length=3)
    questions_to_ask: list[str] = Field(min_length=4, max_length=4)
    preparation_checklist: list[str] = Field(min_length=5, max_length=5)


class InterviewPreparationRecord(BaseModel):
    preparation_id: str
    application_id: str
    provider: str
    model: str
    content: InterviewPreparationContent
    pdf_path: str
    created_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _safe_text(value: Any) -> str:
    """Make model text safe for ReportLab's XML-like Paragraph markup."""

    text = str(value or "")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u2022": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return escape(text)


def _filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip()).strip("-")
    return (cleaned or "interview")[:50].lower()


def _question_story(
    question: InterviewQuestion,
    styles,
    number: int | None = None,
) -> list[Any]:
    evidence = "<br/>".join(
        f"- {_safe_text(item)}" for item in question.cv_evidence
    )
    details = []
    if question.competency:
        details.append(
            Paragraph(
                f"<b>What this tests:</b> {_safe_text(question.competency)}",
                styles["BodyText"],
            )
        )
    details.append(
        Paragraph(
            f"<b>Why it may be asked:</b> {_safe_text(question.why_asked)}",
            styles["BodyText"],
        )
    )
    if question.job_connection:
        details.append(
            Paragraph(
                f"<b>Job connection:</b> {_safe_text(question.job_connection)}",
                styles["BodyText"],
            )
        )
    details.extend(
        [
            Paragraph(
                f"<b>Answer strategy:</b> {_safe_text(question.answer_strategy)}",
                styles["BodyText"],
            ),
            Paragraph(
                f"<b>Practice answer:</b> {_safe_text(question.sample_answer)}",
                styles["BodyText"],
            ),
            Paragraph(f"<b>CV evidence:</b><br/>{evidence}", styles["Evidence"]),
        ]
    )
    if question.learning_plan:
        details.append(
            Paragraph(
                f"<b>Learning plan:</b> {_safe_text(question.learning_plan)}",
                styles["Evidence"],
            )
        )
    return [
        CondPageBreak(80 * mm),
        Paragraph(
            _safe_text(
                f"{number}. {question.question}" if number else question.question
            ),
            styles["Question"],
        ),
        *details,
        Spacer(1, 4 * mm),
    ]


def _plain_list_table(items: list[str], marker: str, styles) -> Table:
    rows = [
        [Paragraph(_safe_text(marker), styles["BodyText"]),
         Paragraph(_safe_text(item), styles["BodyText"])]
        for item in items
    ]
    table = Table(rows, colWidths=[8 * mm, 142 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def render_interview_preparation_pdf(
    *,
    preparation_id: str,
    application: Any,
    content: InterviewPreparationContent,
    output_directory: Optional[str | Path] = None,
) -> Path:
    """Render one validated preparation pack and return its absolute path."""

    directory = Path(output_directory or DEFAULT_INTERVIEW_PDF_DIRECTORY).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"interview-preparation-{_filename_part(application.company)}-"
        f"{_filename_part(application.job_title)}-{preparation_id[:8]}.pdf"
    )
    path = directory / filename

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCentered",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#146C94"),
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Question",
            parent=styles["Heading3"],
            textColor=colors.HexColor("#17324D"),
            spaceAfter=2 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Evidence",
            parent=styles["BodyText"],
            backColor=colors.HexColor("#EEF6FA"),
            borderPadding=6,
            spaceBefore=2 * mm,
        )
    )

    def draw_page(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5C6B73"))
        canvas.drawString(18 * mm, 12 * mm, "Next Chapter - Interview Preparation")
        canvas.drawRightString(
            A4[0] - 18 * mm,
            12 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"Interview Preparation - {application.job_title}",
        author="Next Chapter",
    )
    story: list[Any] = [
        Paragraph("Interview Preparation Guide", styles["TitleCentered"]),
        Paragraph(
            f"<b>{_safe_text(application.job_title)}</b> at "
            f"{_safe_text(application.company)}",
            styles["Heading2"],
        ),
    ]
    summary_data = [
        ["Candidate", application.candidate_name or "Not specified"],
        ["Final match", f"{application.final_score or 0:.1f}%"],
        ["Skills", f"{application.skills_score or 0:.1f}%"],
        ["Experience", f"{application.experience_score or 0:.1f}%"],
        ["Education", f"{application.education_score or 0:.1f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[35 * mm, 115 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F6F8FA")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 3 * mm),
            summary_table,
            Spacer(1, 3 * mm),
            Paragraph(
                f'<link href="{escape(application.url, quote=True)}">'
                "Open the LinkedIn job posting</link>",
                styles["BodyText"],
            ),
            Paragraph("1. Role Snapshot", styles["Section"]),
            Paragraph(_safe_text(content.role_summary), styles["BodyText"]),
        ]
    )
    snapshot = []
    if content.role_family:
        snapshot.append(
            [
                Paragraph("Role family", styles["BodyText"]),
                Paragraph(_safe_text(content.role_family), styles["BodyText"]),
            ]
        )
    if content.primary_focus:
        snapshot.append(
            [
                Paragraph("Primary focus", styles["BodyText"]),
                Paragraph(_safe_text(content.primary_focus), styles["BodyText"]),
            ]
        )
    if snapshot:
        snapshot_table = Table(snapshot, colWidths=[35 * mm, 115 * mm])
        snapshot_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#17324D")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([Spacer(1, 2 * mm), snapshot_table])
    for heading, items in (
        ("Key competencies", content.key_competencies),
        ("Strongest alignments", content.candidate_strengths),
        ("Primary gaps", content.primary_gaps),
    ):
        if items:
            story.extend(
                [
                    Paragraph(heading, styles["Heading3"]),
                    _plain_list_table(items, "-", styles),
                ]
            )
    story.append(Paragraph("2. Technical Interview Questions", styles["Section"]))
    for index, question in enumerate(content.technical_questions, start=1):
        story.extend(_question_story(question, styles, index))

    story.append(Paragraph("3. Gap Questions", styles["Section"]))
    for index, question in enumerate(content.gap_questions, start=1):
        story.extend(_question_story(question, styles, index))

    story.append(Paragraph("4. Behavioral Questions", styles["Section"]))
    for index, question in enumerate(content.behavioral_questions, start=1):
        story.extend(_question_story(question, styles, index))

    story.append(Paragraph("5. Questions to Ask the Interviewer", styles["Section"]))
    story.append(_plain_list_table(content.questions_to_ask, "-", styles))

    story.append(Paragraph("6. Final Preparation Checklist", styles["Section"]))
    story.append(_plain_list_table(content.preparation_checklist, "[ ]", styles))

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return path


def _candidate_prompt_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Exclude contact identifiers before sending profile evidence to the LLM."""

    skills = [str(item)[:120] for item in (profile.get("skills") or [])[:40]]
    evidence = profile.get("skill_evidence") or {}
    return {
        "full_name": str(profile.get("full_name") or "")[:120],
        "skills": skills,
        "skill_evidence": {
            skill: str(evidence.get(skill) or "")[:400]
            for skill in skills
            if evidence.get(skill)
        },
        "job_titles": [
            str(item)[:160] for item in (profile.get("job_titles") or [])[:10]
        ],
        "experience_years": profile.get("experience_years"),
        "education": [
            str(item)[:300] for item in (profile.get("education") or [])[:10]
        ],
        "highest_education_level": str(
            profile.get("highest_education_level") or ""
        )[:80],
    }


def _upgrade_stored_content(payload: dict[str, Any]) -> dict[str, Any]:
    """Add fields introduced after older preparation records were saved."""

    upgraded = dict(payload)
    upgraded.setdefault("role_family", "Not recorded")
    upgraded.setdefault("primary_focus", "See the role summary below.")
    upgraded.setdefault("key_competencies", [])
    upgraded.setdefault("candidate_strengths", [])
    upgraded.setdefault("primary_gaps", [])
    for section in (
        "technical_questions",
        "gap_questions",
        "behavioral_questions",
    ):
        questions = []
        for item in upgraded.get(section, []):
            question = dict(item)
            question.setdefault("competency", "Not recorded")
            question.setdefault("job_connection", "")
            question.setdefault("learning_plan", "")
            questions.append(question)
        upgraded[section] = questions
    return upgraded


def _record_from_row(row: Any) -> InterviewPreparationRecord:
    envelope = json.loads(row["content_json"])
    return InterviewPreparationRecord(
        preparation_id=row["id"],
        application_id=row["application_id"],
        provider=envelope["provider"],
        model=envelope["model"],
        content=InterviewPreparationContent.model_validate(
            _upgrade_stored_content(envelope["content"])
        ),
        pdf_path=envelope["pdf_path"],
        created_at=row["created_at"],
    )


def _default_questions_to_ask(application: Any) -> list[str]:
    return [
        f"What would success in the {application.job_title} role look like after 90 days?",
        f"Which projects would this role contribute to first at {application.company}?",
        "How does the team measure quality, reliability, and business impact?",
        "What are the largest technical challenges the person in this role will own?",
    ]


def _default_preparation_checklist(application: Any) -> list[str]:
    return [
        f"Review the {application.job_title} description and identify the top three requirements.",
        "Prepare two CV-backed project examples with measurable outcomes.",
        "Practice concise explanations of the strongest matching skills.",
        "Prepare honest learning plans for the identified skill gaps.",
        "Confirm the interview format and prepare questions for the interviewer.",
    ]


def _recover_schema_failed_generation(
    error: Exception,
    application: Any,
) -> Optional[InterviewPreparationContent]:
    """Recover a usable Groq payload when only trailing required lists are absent."""

    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    error_body = body.get("error", body)
    if not isinstance(error_body, dict):
        return None
    failed_generation = error_body.get("failed_generation")
    if not isinstance(failed_generation, str):
        return None
    try:
        payload = json.loads(failed_generation)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    # These two sections are general interview guidance, so deterministic
    # fallbacks are safer and cheaper than making a second LLM request.
    payload.setdefault("questions_to_ask", _default_questions_to_ask(application))
    payload.setdefault(
        "preparation_checklist",
        _default_preparation_checklist(application),
    )
    try:
        return InterviewPreparationContent.model_validate(payload)
    except Exception:
        return None


def get_interview_model_info() -> dict[str, str]:
    """Return the configured interview provider/model without exposing keys."""

    return {
        "provider": os.getenv("INTERVIEW_PROVIDER", "groq").lower(),
        "model": os.getenv("INTERVIEW_MODEL", "openai/gpt-oss-120b"),
    }


def _role_interview_blueprint(application: Any) -> dict[str, Any]:
    """Suggest distinct interview dimensions for the applied role family."""

    title = str(application.job_title or "").casefold()
    if any(
        marker in title
        for marker in ("data engineer", "analytics engineer", "etl developer")
    ):
        family = "Data Engineering"
        areas = [
            "data modeling and schema evolution",
            "batch or streaming pipeline design",
            "orchestration, idempotency, and backfills",
            "data quality, lineage, and observability",
            "scalability, reliability, and cost trade-offs",
        ]
    elif "data scientist" in title or "data science" in title:
        family = "Data Science"
        areas = [
            "business problem framing and success metrics",
            "statistics, experimentation, and uncertainty",
            "feature engineering and prevention of data leakage",
            "model selection, validation, and error analysis",
            "communicating results and monitoring model impact",
        ]
    elif any(
        marker in title
        for marker in ("machine learning", "ml engineer", "ai engineer")
    ):
        family = "Machine Learning Engineering"
        areas = [
            "training data and evaluation design",
            "model architecture and performance trade-offs",
            "serving, latency, and scalable inference",
            "MLOps, monitoring, drift, and reproducibility",
            "failure handling and responsible AI safeguards",
        ]
    elif any(marker in title for marker in ("devops", "platform", "site reliability")):
        family = "Platform and Reliability Engineering"
        areas = [
            "infrastructure design and automation",
            "deployment strategy and rollback safety",
            "observability, incident response, and SLOs",
            "security, permissions, and secrets management",
            "capacity, resilience, and cost trade-offs",
        ]
    elif any(
        marker in title
        for marker in ("software", "backend", "frontend", "full stack", "developer")
    ):
        family = "Software Engineering"
        areas = [
            "system and API design",
            "data structures and implementation trade-offs",
            "testing, debugging, and code quality",
            "performance, security, and reliability",
            "delivery, monitoring, and technical collaboration",
        ]
    else:
        family = "Role-specific"
        areas = [
            "the role's primary domain responsibility",
            "an end-to-end scenario from the job description",
            "a difficult decision and its trade-offs",
            "quality, validation, and failure handling",
            "communication with the role's main stakeholders",
        ]
    return {"role_family": family, "technical_focus_areas": areas}


def build_interview_prompt(application: Any, profile: dict[str, Any]) -> str:
    """Build the single grounded generation prompt shared by every caller."""

    matching = application.match_details.get("matching", [])[:30]
    missing = application.match_details.get("missing", [])[:30]
    blueprint = _role_interview_blueprint(application)
    return f"""You are an expert technical interview coach creating structured content for one candidate-facing Interview Preparation Guide.

SOURCE OF TRUTH
Use only the Candidate Profile, Job, Match Evidence, and Role Interview Blueprint below for candidate-specific facts. Never invent employers, projects, achievements, metrics, technologies, certifications, responsibilities, education, experience, or skills. A missing requirement must remain a gap. When direct evidence is absent, connect honest adjacent evidence to a credible learning plan instead of fabricating experience.

ROLE SNAPSHOT
Infer the role family and primary interview focus from the job title and responsibilities. The actual job description takes priority over the blueprint. Fill role_family, primary_focus, and a role_summary under 100 words. Fill key_competencies from the job, candidate_strengths only from matching evidence, and primary_gaps only from missing evidence.

TECHNICAL QUESTIONS
Create exactly 5 distinct questions for this job. Prefer realistic scenarios, design decisions, debugging, trade-offs, validation, reliability, and practical problem solving. At least 3 job_connection fields must cite a recognizable responsibility or requirement from the posting. Cover the five blueprint areas without testing the same concept twice. Avoid questions reusable unchanged for unrelated jobs. For every question, fill competency, why_asked, job_connection, answer_strategy, cv_evidence, and a concise first-person sample_answer.

For Data Science, cover problem framing, statistics or experimentation, leakage, model evaluation, and communication of results. For Data Engineering, cover data modeling, batch or streaming pipelines, orchestration and backfills, data quality, and scale or reliability. For another field, infer five equivalent dimensions from the posting.

GAP AND BEHAVIORAL QUESTIONS
Create exactly 2 gap questions tied to genuine missing or weak requirements. If Match Evidence contains fewer than two missing items, use explicit job requirements that the profile does not demonstrate and label them "not demonstrated" rather than claiming the candidate lacks the skill. Never invent a requirement. Name the gap or evidence uncertainty in job_connection, connect adjacent CV evidence when present, keep the sample answer honest, and provide a concrete learning_plan. Create exactly 3 behavioral questions tied to this job's responsibilities or working patterns. Use STAR only when the profile supports it; otherwise state which detail the candidate should prepare.

FINAL SECTIONS
Create exactly 4 thoughtful, role-specific questions_to_ask about expectations, challenges, priorities, collaboration, or success. Do not assume the company is an AI company. Create exactly 5 concrete preparation_checklist items prioritized for this candidate and job.

OUTPUT RULES
Return exactly one complete object matching the supplied schema. Include every field. Keep why_asked under 25 words, answer_strategy under 35 words, and sample_answer under 70 words. Use no more than two short cv_evidence items per question. Silently verify the item counts, job connections, first-person answers, grounding, and gap honesty before returning.

ROLE INTERVIEW BLUEPRINT:
{json.dumps(blueprint, ensure_ascii=False)}

CANDIDATE PROFILE:
{json.dumps(_candidate_prompt_profile(profile), ensure_ascii=False)}

JOB:
{json.dumps({"title": application.job_title, "company": application.company, "description": str(application.description or "")[:12000]}, ensure_ascii=False)}

MATCH EVIDENCE:
{json.dumps({"matching": matching, "missing": missing, "final_score": application.final_score}, ensure_ascii=False)}
"""


def _is_schema_generation_error(error: Exception) -> bool:
    """Identify provider or local validation failures that are safe to retry."""

    if isinstance(error, ValidationError):
        return True
    body = getattr(error, "body", None)
    error_body = body.get("error", body) if isinstance(body, dict) else {}
    code = str(error_body.get("code") or "").casefold()
    message = str(error_body.get("message") or error).casefold()
    return code in {"json_validate_failed", "tool_use_failed"} or any(
        marker in message
        for marker in ("failed to validate json", "invalid tool call")
    )


def _invoke_structured_interview(llm: Any, prompt: str) -> InterviewPreparationContent:
    structured_llm = llm.with_structured_output(
        InterviewPreparationContent,
        method="json_schema",
        strict=True,
    )
    response = structured_llm.invoke(prompt)
    return (
        response
        if isinstance(response, InterviewPreparationContent)
        else InterviewPreparationContent.model_validate(response)
    )


def _fallback_interview_content(
    application: Any,
    profile: dict[str, Any],
) -> InterviewPreparationContent:
    """Build a complete, honest pack if repeated provider JSON generation fails."""

    details = application.match_details or {}
    matching = details.get("matching") or []
    missing = [
        str(item)[:160]
        for item in (details.get("missing") or [])
        if str(item)
    ]
    skills = [
        str(item)[:160] for item in (profile.get("skills") or []) if str(item)
    ]
    evidence_map = profile.get("skill_evidence") or {}
    blueprint = _role_interview_blueprint(application)

    matched_topics: list[str] = []
    matched_evidence: dict[str, str] = {}
    for item in matching:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("job_skill") or "").strip()[:160]
        if not topic or topic in matched_topics:
            continue
        matched_topics.append(topic)
        via = str(item.get("matched_via") or topic).strip()
        matched_evidence[topic] = str(
            evidence_map.get(via) or evidence_map.get(topic) or f"CV lists {via}."
        )

    topics = list(blueprint["technical_focus_areas"])
    topics.extend(matched_topics)
    topics.extend(skill for skill in skills if skill not in matched_topics)
    unique_topics = list(dict.fromkeys(topics))[:5]

    def evidence_for(topic: str) -> str:
        return str(
            matched_evidence.get(topic)
            or evidence_map.get(topic)
            or (
                f"CV lists {topic}."
                if topic in skills
                else "No direct CV evidence; prepare an honest learning example."
            )
        )[:500]

    question_stems = [
        "Walk me through how you would approach {topic} for this role.",
        "How would you design a solution for {topic}, and what trade-offs would you make?",
        "What can go wrong with {topic}, and how would you diagnose it?",
        "How would you validate {topic} before relying on the result?",
        "What trade-offs would you consider when improving {topic}?",
    ]
    technical = [
        InterviewQuestion(
            question=question_stems[index].format(topic=topic),
            competency=topic,
            why_asked=(
                f"This checks practical {blueprint['role_family']} judgment in a "
                "relevant area."
            ),
            job_connection=(
                f"The {application.job_title} interview may evaluate {topic}."
            ),
            answer_strategy=(
                "Explain the problem, approach, trade-offs, validation, and result using "
                "only experience supported by the CV."
            ),
            sample_answer=(
                f"My relevant evidence is: {evidence_for(topic)} I would explain the "
                "problem I faced, the choices I made, and how I verified the result."
            ),
            cv_evidence=[evidence_for(topic)],
            learning_plan="",
        )
        for index, topic in enumerate(unique_topics)
    ]

    gap_topics = (missing + ["an unfamiliar requirement", "a new team tool"])[:2]
    gaps = [
        InterviewQuestion(
            question=f"How would you close your experience gap in {topic}?",
            competency="Honest gap management and learning agility",
            why_asked="This checks honesty, learning speed, and practical planning.",
            job_connection=f"Identified missing or weak requirement: {topic}.",
            answer_strategy=(
                "Acknowledge the gap, connect adjacent CV evidence, and give a concrete "
                "learning and validation plan."
            ),
            sample_answer=(
                f"My CV does not claim direct experience with {topic}. I would be clear "
                "about that, build on my related skills, and validate my learning with a "
                "small practical task and feedback from the team."
            ),
            cv_evidence=["No direct CV evidence; this is an identified job gap."],
            learning_plan=(
                f"Complete a focused {topic} learning task, build a small example, "
                "and request feedback before the interview."
            ),
        )
        for topic in gap_topics
    ]

    primary_evidence = evidence_for(unique_topics[0])
    behavioral_prompts = [
        "Tell me about a difficult technical problem you worked through.",
        "Describe a time you had to learn a new technology quickly.",
        "Tell me about a project where you collaborated with other people.",
    ]
    behavioral = [
        InterviewQuestion(
            question=question,
            competency="Communication, ownership, and reflective problem solving",
            why_asked="This checks communication, ownership, and evidence-based reflection.",
            job_connection=(
                f"Behavioral preparation for responsibilities in the "
                f"{application.job_title} role."
            ),
            answer_strategy=(
                "Use Situation, Task, Action, and Result. Choose a CV-backed example and "
                "avoid adding unsupported metrics."
            ),
            sample_answer=(
                f"I would use this CV evidence: {primary_evidence} I would state the "
                "situation and my responsibility, describe my actions, and finish with "
                "the verified result and what I learned."
            ),
            cv_evidence=[primary_evidence],
            learning_plan="",
        )
        for question in behavioral_prompts
    ]

    return InterviewPreparationContent(
        role_summary=(
            "Structured model generation was unavailable, so this recovery pack "
            f"prepares for the {application.job_title} role at {application.company} "
            f"with a {blueprint['role_family']} focus, verified CV evidence, "
            "role-specific technical judgment, and honest plans for skill gaps."
        ),
        role_family=blueprint["role_family"],
        primary_focus=", ".join(blueprint["technical_focus_areas"][:2]),
        key_competencies=list(blueprint["technical_focus_areas"]),
        candidate_strengths=matched_topics[:6],
        primary_gaps=missing[:6],
        technical_questions=technical,
        gap_questions=gaps,
        behavioral_questions=behavioral,
        questions_to_ask=_default_questions_to_ask(application),
        preparation_checklist=_default_preparation_checklist(application),
    )


def generate_interview_content(
    application: Any,
    profile: dict[str, Any],
    *,
    llm: Any = None,
) -> InterviewPreparationContent:
    """Run the structured interview LLM call only; no files or rows are written.

    This is the step Agent 2's interview graph and Agent 3's tool both rely
    on, so the prompt, schema enforcement, and Groq schema-recovery fallback
    live here exactly once.
    """

    prompt = build_interview_prompt(application, profile)
    llm_was_provided = llm is not None
    if not llm_was_provided:
        llm = get_interview_llm(temperature=0.2)
    try:
        return _invoke_structured_interview(llm, prompt)
    except Exception as error:
        recovered = _recover_schema_failed_generation(error, application)
        if recovered is not None:
            return recovered
        if not _is_schema_generation_error(error):
            raise

    retry_llm = llm if llm_was_provided else get_interview_llm(temperature=0.0)
    retry_prompt = (
        prompt
        + "\nRETRY: The previous response failed JSON schema validation. Return one "
        "complete object only, with every required list and exact item count."
    )
    try:
        return _invoke_structured_interview(retry_llm, retry_prompt)
    except Exception as error:
        recovered = _recover_schema_failed_generation(error, application)
        if recovered is not None:
            return recovered
        if not _is_schema_generation_error(error):
            raise
        return _fallback_interview_content(application, profile)


def persist_interview_preparation(
    *,
    preparation_id: str,
    application_id: str,
    content: InterviewPreparationContent,
    pdf_path: str | Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    database_path: Optional[str | Path] = None,
) -> InterviewPreparationRecord:
    """Store one rendered preparation version and return its record."""

    model_info = get_interview_model_info()
    provider = provider or model_info["provider"]
    model = model or model_info["model"]
    envelope = {
        "provider": provider,
        "model": model,
        "content": content.model_dump(mode="json"),
        "pdf_path": str(pdf_path),
    }
    created_at = _utc_now()
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        connection.execute(
            """INSERT INTO interview_preparations(
                   id, application_id, content_json, created_at
               ) VALUES (?, ?, ?, ?)""",
            (
                preparation_id,
                application_id,
                json.dumps(envelope, ensure_ascii=False),
                created_at,
            ),
        )
    return InterviewPreparationRecord(
        preparation_id=preparation_id,
        application_id=application_id,
        provider=provider,
        model=model,
        content=content,
        pdf_path=str(pdf_path),
        created_at=created_at,
    )


def generate_interview_preparation(
    application_id: str,
    *,
    llm: Any = None,
    output_directory: Optional[str | Path] = None,
    database_path: Optional[str | Path] = None,
) -> InterviewPreparationRecord:
    """Generate one grounded pack, render its PDF, and persist its metadata.

    Compatibility wrapper that runs the three shared steps in order. Agent 2's
    interview graph calls the same steps as separate nodes; Agent 3's tool
    calls this wrapper directly.
    """

    application = get_application(application_id, database_path=database_path)
    profile = get_candidate_profile(
        application.candidate_id,
        database_path=database_path,
    )
    content = generate_interview_content(application, profile, llm=llm)
    preparation_id = str(uuid4())
    pdf_path = render_interview_preparation_pdf(
        preparation_id=preparation_id,
        application=application,
        content=content,
        output_directory=output_directory,
    )
    return persist_interview_preparation(
        preparation_id=preparation_id,
        application_id=application.application_id,
        content=content,
        pdf_path=pdf_path,
        database_path=database_path,
    )


def list_interview_preparations(
    application_id: str,
    *,
    database_path: Optional[str | Path] = None,
) -> list[InterviewPreparationRecord]:
    """Return all preparation versions for one application, newest first."""

    resolved_id = str(application_id or "").strip()
    if not resolved_id:
        raise ValueError("application_id is required.")
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        rows = connection.execute(
            """SELECT id, application_id, content_json, created_at
               FROM interview_preparations
               WHERE application_id = ?
               ORDER BY created_at DESC""",
            (resolved_id,),
        ).fetchall()
    return [_record_from_row(row) for row in rows]

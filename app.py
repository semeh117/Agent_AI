"""Local job-search workspace. Run: python -m streamlit run app.py"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st

from agent.agent2 import (get_agent2_workflow, resume_agent2_workflow,
                          retry_agent2_delivery, run_agent2_full_auto)
from agent.agent2_interview import run_agent2_interview_preparation
from core.agent2_cv_parser import Agent2CVInfo
from services.application_tracker import (
    add_application_note, get_application_status_history, get_candidate_profile,
    get_tracker_summary, list_applications, list_candidates, update_application_status,
)
from services.interview_preparation import list_interview_preparations
from services.ui_support import parse_uploaded_cv, provider_ready, sample_profile, sample_result
from storage.agent2_database import AGENT2_APPLICATION_STATUSES

st.set_page_config(page_title="Next Chapter · Job workspace", page_icon="🌱", layout="wide")
st.markdown("""<style>
.block-container {max-width:1200px;padding-top:2.4rem;padding-bottom:3rem;}
h1 {letter-spacing:-.045em;font-weight:650!important;}
h2,h3 {letter-spacing:-.025em;}
[data-testid="stMetric"] {padding:1rem;border:1px solid #dedfd6;border-radius:12px;}
[data-testid="stSidebar"] {border-right:1px solid #dedfd6;}
div.stButton > button {border-radius:8px;}
</style>""", unsafe_allow_html=True)


def remember_result(result: dict):
    st.session_state.result = result
    if result.get("workflow_id") != "sample":
        st.query_params["workflow"] = result["workflow_id"]


def load_profile(profile: dict):
    st.session_state.profile = profile
    st.session_state.profile_revision = st.session_state.get("profile_revision", 0) + 1
    st.session_state.profile_confirmed = False


def show_error(exc):
    st.error(str(exc))


def profile_page():
    st.caption("01 / YOUR STARTING POINT")
    st.title("Your experience. Your next chapter.")
    st.write("Turn your CV into a focused search, understand your matches, and keep every application in one place.")
    if "profile" not in st.session_state:
        columns = st.columns(3)
        for column, title, description in zip(columns,
                ["Review your profile", "Find your fit", "Move forward"],
                ["Check the skills and experience extracted from your CV.",
                 "See which requirements you meet and where the gaps are.",
                 "Prepare your letter, track progress, and practise for interviews."]):
            with column.container(border=True):
                st.subheader(title)
                st.write(description)
        if st.button("Explore a sample workspace", icon="🌱"):
            load_profile(sample_profile())
            remember_result(sample_result())
            st.session_state.sample = True
            st.session_state.page = "Job matches"
            st.rerun()
    with st.container(border=True):
        st.subheader("Start with your CV")
        st.caption("PDF · up to 10 MB. Parsed profiles and workflow history stay in this local workspace. "
                   "Text is sent to your configured model providers when you parse or generate content.")
        upload = st.file_uploader("Upload your CV", type=["pdf"])
        if not provider_ready("parser"):
            st.info("Add your parser API key to .env to extract a CV. You can explore the sample without an API key.")
        if st.button("Extract my profile", type="primary", disabled=upload is None or not provider_ready("parser")):
            try:
                with st.spinner("Reading your CV and extracting your profile…"):
                    profile, warnings = parse_uploaded_cv(upload.getvalue())
                load_profile(profile)
                st.session_state.sample = False
                st.session_state.pop("result", None)
                st.query_params.clear()
                for warning in warnings:
                    st.warning(warning)
            except Exception as exc:
                show_error(exc)
    if "profile" not in st.session_state:
        return
    profile = st.session_state.profile
    st.subheader("Review before you search")
    st.caption("Correct anything that was missed. Only include skills and experience you can support.")
    with st.form(f"profile_{st.session_state.get('profile_revision', 0)}"):
        left, right = st.columns(2)
        name = left.text_input("Full name", profile.get("full_name") or "")
        email = right.text_input("Email", profile.get("mail") or "")
        headline = left.text_input("Target role / headline", profile.get("headline") or "")
        years = right.number_input("Years of experience", min_value=0.0, max_value=70.0,
                                   value=float(profile.get("experience_years") or 0), step=0.25)
        titles = left.text_area("Previous roles (one per line)", "\n".join(profile.get("job_titles", [])))
        levels = ["Not specified", "High School", "Bachelor", "Master", "PhD"]
        education = right.selectbox("Education level", levels,
            index=levels.index(profile.get("highest_education_level")) if profile.get("highest_education_level") in levels else 0)
        degrees = right.text_area("Degrees / programmes (one per line)", "\n".join(profile.get("education", [])))
        skills = st.text_area("Skills (one per line)", "\n".join(profile.get("skills", [])), height=180)
        if st.form_submit_button("Save reviewed profile", type="primary"):
            split = lambda value: list(dict.fromkeys(line.strip() for line in value.splitlines() if line.strip()))
            if not name.strip() or not split(skills):
                st.error("Enter your name and at least one skill.")
            else:
                corrected = {**profile, "full_name": name.strip(), "mail": email.strip() or None,
                    "headline": headline.strip() or None, "experience_years": years,
                    "job_titles": split(titles), "skills": split(skills), "education": split(degrees),
                    "highest_education_level": None if education == "Not specified" else education}
                st.session_state.profile = Agent2CVInfo.model_validate(corrected).model_dump()
                st.session_state.profile_confirmed = True
                st.success("Profile saved. Open Job matches to start your search.")


def match_card(job: dict, rank: int):
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.caption(f"MATCH {rank:02d} · {job.get('company', '')}")
        left.subheader(job.get("job_title", "Untitled role"))
        right.metric("Compatibility", f"{job.get('final_score', 0):.0f}%")
        if job.get("inconclusive"):
            st.warning("Insufficient requirements were extracted. This score is inconclusive.")
        st.caption(f"Skills {job.get('skills_score', 0):.0f}%  ·  Experience {job.get('experience_score', 0):.0f}%  ·  Education {job.get('education_score', 0):.0f}%")
        with st.expander("Why this match?"):
            details = job.get("skills_detail", {})
            left, right = st.columns(2)
            left.markdown("**Evidence found**")
            for item in details.get("matching", []):
                left.write(f"✓ {item['job_skill']} — {item.get('matched_via', '')}")
            right.markdown("**Not found in your profile**")
            for item in details.get("missing", []):
                right.write(f"• {item}")
            st.caption("Scores reflect extracted requirements, not hiring probability. Unspecified experience "
                       "and education receive no penalty. Explicit alternatives count as one requirement.")
            st.write(job.get("description", ""))
        url = job.get("url", "")
        if url.startswith(("https://", "http://")):
            st.link_button("View job posting ↗", url)


def delivery_panel(result):
    st.subheader("Your cover letter")
    if result.get("status") == "sample":
        st.info("Sample content with illustrative scores. Live search uses your reviewed profile.")
        st.text_area("Sample letter", result.get("cover_letter", ""), height=240)
        return
    letter = result.get("cover_letter", "")
    if not letter:
        st.info("No cover letter is available for this run. Your saved matches remain in Applications.")
        return
    with st.form(f"delivery_{result['workflow_id']}"):
        edited = st.text_area("Review and edit your letter", letter, height=280,
                              disabled=result.get("status") != "awaiting_delivery")
        channel = st.radio("Delivery", ["Gmail draft", "Telegram message"], horizontal=True)
        st.caption("Gmail creates a draft addressed to your CV email. Telegram sends to the chat configured in .env.")
        approved = st.form_submit_button("Approve and deliver", type="primary",
                                         disabled=result.get("status") != "awaiting_delivery")
    st.download_button("Download letter", edited, file_name="cover-letter.txt", mime="text/plain")
    if approved:
        try:
            with st.spinner("Delivering your reviewed results…"):
                updated = resume_agent2_workflow(result["workflow_id"],
                    "gmail" if channel == "Gmail draft" else "telegram", cover_letter=edited)
            remember_result(updated)
            st.rerun()
        except Exception as exc:
            show_error(exc)
    delivery = result.get("delivery", {})
    if delivery.get("status") == "completed":
        st.success("Gmail draft created." if delivery["channel"] == "gmail" else "Telegram delivery completed.")
    elif delivery.get("status") == "failed":
        st.error(delivery.get("error", "Delivery failed."))
        if delivery.get("channel") == "telegram":
            from services.delivery_journal import pending_parts, resolve_part
            operation = f"telegram:{result['workflow_id']}"
            for part in pending_parts(operation):
                with st.form(f"resolve_{result['workflow_id']}_{part}"):
                    st.warning(f"Check your Telegram chat for message part {part + 1} before retrying.")
                    received = st.radio("Did this part arrive?", ["Yes, it arrived", "No, it did not arrive"], key=f"part_{part}")
                    confirmed = st.checkbox("I checked the recipient's chat", key=f"checked_{part}")
                    if st.form_submit_button("Record delivery outcome") and confirmed:
                        resolve_part(operation, part, received == "Yes, it arrived")
                        st.rerun()
        if st.button("Retry approved delivery"):
            try:
                with st.spinner("Retrying delivery…"):
                    remember_result(retry_agent2_delivery(result["workflow_id"]))
                st.rerun()
            except Exception as exc:
                show_error(exc)


def matches_page():
    st.caption("02 / FIND YOUR FIT")
    st.title("A more focused job search.")
    st.write("Search a wider pool, then bring the strongest matches to the top.")
    profile = st.session_state.get("profile")
    with st.expander("Search preferences", expanded="result" not in st.session_state):
        with st.form("search"):
            query = st.text_input("Target role or search query (optional)", placeholder="e.g. Junior Python developer")
            location = st.text_input("Location", placeholder="City, country, or leave blank")
            left, right = st.columns(2)
            count = left.slider("Recommendations to show", 1, 10, 3)
            pool = right.slider("Job postings to consider", 10, 50, 10, step=5)
            st.caption("Larger searches take longer and use more parser calls. Searches cover the last 30 days.")
            ready = bool(profile and st.session_state.get("profile_confirmed"))
            if not ready:
                st.info("Review and save your profile before starting a live search.")
            if not provider_ready("parser") or not provider_ready("cover_letter"):
                st.info("Configure the parser and cover-letter provider keys in .env to search.")
            search = st.form_submit_button("Find my matches", type="primary", disabled=not ready or
                    not provider_ready("parser") or not provider_ready("cover_letter"))
        if search:
            workflow_id = str(uuid4())
            st.query_params["workflow"] = workflow_id
            try:
                with st.status("Searching for your next role…", expanded=True) as progress:
                    labels = {"load_cv": "Profile loaded", "build_query": "Search query ready",
                        "match_jobs": "Job pool parsed and ranked", "persist_recommendations": "Matches saved",
                        "generate_cover_letter": "Cover letter ready", "finalize": "Search finished"}
                    result = run_agent2_full_auto(Agent2CVInfo.model_validate(profile),
                        results_count=count, location=location, query=query, search_pool_size=pool,
                        interactive_delivery=False, workflow_id=workflow_id,
                        on_progress=lambda node: st.write(labels.get(node, node.replace("_", " ").capitalize())))
                    progress.update(label="Matches ready" if result.get("ranked_jobs") else "Search finished",
                                    state="error" if result.get("error") else "complete", expanded=False)
                st.session_state.sample = False
                remember_result(result)
            except Exception as exc:
                show_error(exc)
    result = st.session_state.get("result")
    if not result:
        st.info("Your ranked matches will appear here after a search.")
        return
    if result.get("error"):
        st.error(result["error"])
    for warning in result.get("warnings", []):
        st.warning(warning)
    jobs = result.get("ranked_jobs", [])
    info = result.get("match_result", {})
    columns = st.columns(3)
    columns[0].metric("Matches", len(jobs))
    columns[1].metric("Jobs evaluated", info.get("parsed_count", 0))
    columns[2].metric("Skipped", info.get("skipped_count", 0))
    if info.get("skipped_jobs"):
        with st.expander("Why some postings were skipped"):
            for job in info["skipped_jobs"]:
                st.write(f"{job.get('title', 'Posting')}: {job['reason']}")
    if not jobs:
        st.info("No matches were found. Try a broader role or location. Previously tracked jobs are excluded.")
        return
    for index, job in enumerate(jobs, 1):
        match_card(job, index)
    delivery_panel(result)


def applications_page():
    st.caption("03 / KEEP MOVING")
    st.title("Every opportunity, in one place.")
    st.write("Track your next action and prepare for the conversations ahead.")
    candidates = list_candidates()
    if not candidates:
        st.info("Your live recommendations will be saved here automatically after a search.")
        return
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_id = st.selectbox("Candidate", list(candidates_by_id), format_func=lambda key:
        f"{candidates_by_id[key].full_name or 'Unnamed'} · {candidates_by_id[key].email or 'No email'}")
    if st.button("Use this saved profile for a new search"):
        load_profile(get_candidate_profile(candidate_id))
        st.session_state.page = "Your profile"
        st.rerun()
    summary = get_tracker_summary(candidate_id)
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Tracked", "Applied", "Interviews", "Offers"],
            [summary.total_applications, summary.status_counts["applied"], summary.status_counts["interview"], summary.status_counts["offer"]]):
        column.metric(label, value)
    left, right = st.columns(2)
    status = left.selectbox("Filter by status", ["All", *AGENT2_APPLICATION_STATUSES])
    search = right.text_input("Search roles or companies")
    applications = list_applications(candidate_id, status=None if status == "All" else status, search=search)
    if not applications:
        st.info("No applications match these filters.")
    for application in applications:
        key = application.application_id
        with st.expander(f"{application.job_title} · {application.company} · {application.status.title()}"):
            if application.url.startswith(("https://", "http://")):
                st.link_button("View posting ↗", application.url)
            with st.form(f"application_{key}"):
                selected = st.selectbox("Application status", AGENT2_APPLICATION_STATUSES,
                    index=AGENT2_APPLICATION_STATUSES.index(application.status))
                note = st.text_area("Add a note", placeholder="Next step, follow-up date, or interview details")
                if st.form_submit_button("Save update"):
                    update_application_status(key, selected)
                    if note.strip():
                        add_application_note(key, note)
                    st.rerun()
            if application.notes:
                st.text(application.notes)
            if application.cover_letter:
                st.download_button("Download saved letter", application.cover_letter,
                    file_name="cover-letter.txt", key=f"letter_{key}")
            history = get_application_status_history(key)
            st.caption(" → ".join(event.new_status.title() for event in history))
            if st.button("Prepare for this interview", key=f"interview_{key}", disabled=not provider_ready("interview")):
                try:
                    with st.spinner("Preparing your interview questions and practice plan…"):
                        result = run_agent2_interview_preparation(key)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.success("Interview preparation saved.")
                except Exception as exc:
                    show_error(exc)
            if not provider_ready("interview"):
                st.caption("Configure your interview provider key in .env to generate a preparation pack.")
            for pack in list_interview_preparations(key):
                path = Path(pack.pdf_path)
                if path.is_file():
                    st.download_button(f"Interview PDF · {pack.created_at[:16]}", path.read_bytes(),
                        file_name=path.name, mime="application/pdf", key=pack.preparation_id)
                else:
                    st.warning("A saved preparation PDF is missing from disk. Generate a new version.")


def main():
    if "result" not in st.session_state and st.query_params.get("workflow"):
        try:
            restored = get_agent2_workflow(st.query_params["workflow"])
            st.session_state.result = restored
            if restored.get("cv_info"):
                load_profile(restored["cv_info"].model_dump())
            st.session_state.page = "Job matches"
        except LookupError:
            st.warning("That saved search could not be found. Start a new search or restore another ID.")
    with st.sidebar:
        st.markdown("### 🌱 Next Chapter")
        st.caption("YOUR JOB SEARCH WORKSPACE")
        page = st.radio("Workspace", ["Your profile", "Job matches", "Applications"], key="page", label_visibility="collapsed")
        st.divider()
        st.caption("LOCAL WORKSPACE")
        st.write("A little clarity for your next move.")
        with st.expander("Restore a saved search"):
            identifier = st.text_input("Workflow ID")
            if st.button("Restore search", disabled=not identifier.strip()):
                try:
                    restored = get_agent2_workflow(identifier.strip())
                    remember_result(restored)
                    if restored.get("cv_info"):
                        load_profile(restored["cv_info"].model_dump())
                    st.rerun()
                except Exception as exc:
                    show_error(exc)
        if st.session_state.get("result", {}).get("workflow_id") not in (None, "sample"):
            st.caption("Saved search ID")
            st.code(st.session_state.result["workflow_id"], language=None)
    try:
        {"Your profile": profile_page, "Job matches": matches_page, "Applications": applications_page}[page]()
    except Exception as exc:
        show_error(exc)


if __name__ == "__main__":
    main()

"""Streamlit applications: extracted from the original app.py."""
import streamlit as st
from pathlib import Path
from next_chapter.agents.agent2_interview import run_agent2_interview_preparation
from next_chapter.services.application_tracker import (
    add_application_note, get_application_status_history, get_candidate_profile,
    get_tracker_summary, list_applications, list_candidates, update_application_status,
)
from next_chapter.services.interview_preparation import list_interview_preparations
from next_chapter.storage.agent2_database import AGENT2_APPLICATION_STATUSES
from next_chapter.ui.support import provider_ready
from next_chapter.ui.components import page_header, navigate, load_profile, show_error


def applications_page():
    page_header(2, "Every opportunity, in one place.",
                "Keep track of your progress, capture your next action, and get ready for the conversations ahead.")
    candidates = list_candidates()
    if not candidates:
        st.subheader("Your next opportunity starts with a search")
        st.write("Matches will appear here automatically. Then you can update their status, save notes, and prepare for interviews.")
        if st.button("Find job matches →", type="primary"):
            navigate("Job matches")
        return
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    current_id = st.session_state.get("result", {}).get("candidate_id")
    candidate_ids = list(candidates_by_id)
    candidate_id = st.selectbox("Candidate", candidate_ids,
        index=candidate_ids.index(current_id) if current_id in candidate_ids else 0, format_func=lambda key:
        f"{candidates_by_id[key].full_name or 'Unnamed'} · {candidates_by_id[key].email or 'No email'}")
    if st.button("Use this saved profile for a new search"):
        load_profile(get_candidate_profile(candidate_id))
        st.session_state.next_page = "Your profile"
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
            st.markdown("**Interview preparation**")
            packs = list_interview_preparations(key)
            st.caption("Download a saved pack below, or generate a new version tailored to this role." if packs else
                       "Create a question pack and practice plan tailored to your profile and this role.")
            if st.button("Generate new version" if packs else "Prepare for this interview",
                         key=f"interview_{key}", disabled=not provider_ready("interview")):
                try:
                    with st.spinner("Preparing your interview questions and practice plan…"):
                        result = run_agent2_interview_preparation(key)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.success("Interview preparation saved.")
                        packs = list_interview_preparations(key)
                except Exception as exc:
                    show_error(exc)
            if not provider_ready("interview"):
                st.caption("Configure your interview provider key in .env to generate a preparation pack.")
            for pack in packs:
                path = Path(pack.pdf_path)
                if path.is_file():
                    st.download_button(f"Interview PDF · {pack.created_at[:16]}", path.read_bytes(),
                        file_name=path.name, mime="application/pdf", key=pack.preparation_id)
                else:
                    st.warning("A saved preparation PDF is missing from disk. Generate a new version.")

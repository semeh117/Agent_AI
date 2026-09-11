"""Streamlit components: extracted from the original app.py."""
import streamlit as st
from html import escape
from next_chapter.agents.agent2 import resume_agent2_workflow, retry_agent2_delivery


def navigate(page):
    st.session_state.next_page = page
    st.rerun()


def page_header(step, title, description):
    st.markdown(f'<div class="workspace-hero"><div class="eyebrow">NEXT CHAPTER / YOUR JOB SEARCH</div>'
                f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)
    labels = ["Your profile", "Job matches", "Applications"]
    steps = ''.join(f'<div class="journey-step{" active" if i == step else ""}"'
                    f'{" aria-current=step" if i == step else ""}><span>0{i+1}</span>{label}</div>'
                    for i, label in enumerate(labels))
    st.markdown(f'<nav class="journey" aria-label="Your job search steps">{steps}</nav>', unsafe_allow_html=True)


def remember_result(result: dict):
    st.session_state.result = result
    st.session_state.sample = result.get("status") == "sample"
    if result.get("workflow_id") != "sample":
        st.query_params["workflow"] = result["workflow_id"]


def load_profile(profile: dict):
    st.session_state.profile = profile
    st.session_state.sample = False
    st.session_state.profile_revision = st.session_state.get("profile_revision", 0) + 1
    st.session_state.profile_confirmed = False


def show_error(exc):
    st.error(str(exc))


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
            from next_chapter.services.delivery_journal import pending_parts, resolve_part
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

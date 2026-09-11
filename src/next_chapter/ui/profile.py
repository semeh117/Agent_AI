"""Streamlit profile: extracted from the original app.py."""
import streamlit as st
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo
from next_chapter.ui.support import parse_uploaded_cv, provider_ready, sample_profile, sample_result
from next_chapter.ui.components import page_header, load_profile, remember_result, navigate, show_error


def profile_page():
    page_header(0, "Your experience. Your next chapter.",
                "Find roles that fit your experience, understand each match, and take your next step with confidence.")
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
            st.session_state.next_page = "Job matches"
            st.rerun()
        st.caption("Just looking around? The sample uses fictional jobs and requires no setup.")
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
                st.success("Your profile is ready. Continue to choose your role and location.")
    if st.session_state.get("profile_confirmed"):
        if st.button("Continue to job matches →", type="primary"):
            navigate("Job matches")

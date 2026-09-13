"""Streamlit app: extracted from the original app.py."""
import streamlit as st
from importlib.resources import files
from next_chapter.agents.agent2 import get_agent2_workflow
from next_chapter.ui.components import load_profile, remember_result, show_error
from next_chapter.ui.profile import profile_page
from next_chapter.ui.matches import matches_page
from next_chapter.ui.applications import applications_page


def main():
    st.set_page_config(page_title="Next Chapter · Job workspace", page_icon="🌱", layout="wide")
    css = files("next_chapter.ui").joinpath("assets/workspace.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if "next_page" in st.session_state:
        st.session_state.page = st.session_state.pop("next_page")
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
        st.markdown('<div class="brand">🌱 Next Chapter</div>', unsafe_allow_html=True)
        st.caption("A LITTLE CLARITY. A NEW BEGINNING.")
        st.divider()
        page = st.radio("Workspace", ["Your profile", "Job matches", "Applications"], key="page", label_visibility="collapsed")
        st.divider()
        st.caption("YOUR WORKSPACE")
        if st.session_state.get("sample"):
            st.write("🌱 Exploring the sample")
        elif st.session_state.get("profile"):
            st.write(st.session_state.profile.get("full_name") or "Your profile")
            st.caption("Profile reviewed · ready to search" if st.session_state.get("profile_confirmed") else "Review your profile before searching")
        else:
            st.write("Start by adding your CV")
            st.caption("Your saved applications stay available here.")
        with st.expander("Restore a saved search"):
            identifier = st.text_input("Workflow ID")
            if st.button("Restore search", disabled=not identifier.strip()):
                try:
                    restored = get_agent2_workflow(identifier.strip())
                    remember_result(restored)
                    if restored.get("cv_info"):
                        load_profile(restored["cv_info"].model_dump())
                    st.session_state.next_page = "Job matches"
                    st.rerun()
                except Exception as exc:
                    show_error(exc)
        current_result = st.session_state.get("result", {})
        if (
            current_result.get("workflow_type", "agent2") == "agent2"
            and current_result.get("workflow_id") not in (None, "sample")
        ):
            st.caption("Saved search ID")
            st.code(current_result["workflow_id"], language=None)
    try:
        {"Your profile": profile_page, "Job matches": matches_page, "Applications": applications_page}[page]()
    except Exception as exc:
        show_error(exc)

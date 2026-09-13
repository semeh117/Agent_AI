"""Streamlit matches: extracted from the original app.py."""
import streamlit as st
from uuid import uuid4
from next_chapter.agents.agent2 import run_agent2_full_auto
from next_chapter.agents.agent3 import run_agent3_full_auto
from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo
from next_chapter.ui.support import provider_ready
from next_chapter.ui.components import (
    delivery_panel,
    match_card,
    navigate,
    page_header,
    react_trace_panel,
    remember_result,
    show_error,
)


def matches_page():
    page_header(
        1,
        "Make your next move a good fit.",
        "Compare your strongest matches, understand the gaps, and prepare a "
        "letter you can make your own.",
    )
    profile = st.session_state.get("profile")
    sample = st.session_state.get("sample", False)
    if sample:
        st.info("You’re exploring a sample workspace. Jobs, scores, and the profile are fictional.")
        if st.button("Start with my own CV", type="primary"):
            for key in ("profile", "profile_confirmed", "result", "sample"):
                st.session_state.pop(key, None)
            st.query_params.clear()
            navigate("Your profile")
    elif profile:
        left, right = st.columns([3, 1])
        left.write(f"Searching for **{profile.get('full_name') or 'you'}** · {len(profile.get('skills', []))} skills in your profile")
        if right.button("Edit profile", use_container_width=True):
            navigate("Your profile")
    else:
        st.info("Start with your CV so your matches reflect your skills and experience.")
        if st.button("Go to your profile →", type="primary"):
            navigate("Your profile")
        return
    with st.expander(
        "Search preferences",
        expanded="result" not in st.session_state,
    ):
        with st.form("search"):
            workflow = st.radio(
                "Search workflow",
                ["Agent 2 · LangGraph", "Agent 3 · ReAct"],
                horizontal=True,
                help=(
                    "Agent 2 follows a checkpointed production graph. Agent 3 "
                    "chooses its next tool from each ReAct observation."
                ),
            )
            query = st.text_input(
                "Target role or search query (optional)",
                placeholder="e.g. Junior Python developer",
            )
            location = st.text_input(
                "Location",
                placeholder="City, country, or leave blank",
            )
            left, right = st.columns(2)
            count = left.slider("Recommendations to show", 1, 10, 3)
            pool = right.slider("Job postings to consider", 1, 10, 3)
            st.caption(
                "Larger searches take longer and use more parser calls. "
                "Searches cover the last 30 days."
            )
            ready = bool(
                profile
                and st.session_state.get("profile_confirmed")
                and not sample
            )
            if not ready:
                st.info("Review and save your profile before starting a live search.")
            agent_role = "agent3" if workflow.startswith("Agent 3") else "agent"
            providers_ready = all(
                provider_ready(role)
                for role in ("parser", "cover_letter", agent_role)
            )
            if not providers_ready:
                st.info(
                    "Configure the parser, workflow-agent, and cover-letter "
                    "provider keys before searching."
                )
            search = st.form_submit_button(
                "Find my matches",
                type="primary",
                disabled=not ready or not providers_ready,
            )
        if search:
            workflow_id = str(uuid4())
            try:
                with st.status("Searching for your next role…", expanded=True) as progress:
                    cv_info = Agent2CVInfo.model_validate(profile)
                    if workflow.startswith("Agent 3"):
                        st.write("Agent 3 is running its ReAct action/observation loop.")
                        result = run_agent3_full_auto(
                            cv_info,
                            results_count=count,
                            location=location,
                            query=query,
                            search_pool_size=max(pool, count),
                            workflow_id=workflow_id,
                            verbose=False,
                        )
                    else:
                        labels = {
                            "load_cv": "Profile loaded",
                            "build_query": "Search query ready",
                            "match_jobs": "Job pool parsed and ranked",
                            "persist_recommendations": "Matches saved",
                            "generate_cover_letter": "Cover letter ready",
                            "finalize": "Search finished",
                        }
                        result = run_agent2_full_auto(
                            cv_info,
                            results_count=count,
                            location=location,
                            query=query,
                            search_pool_size=max(pool, count),
                            interactive_delivery=False,
                            workflow_id=workflow_id,
                            on_progress=lambda node: st.write(
                                labels.get(
                                    node,
                                    node.replace("_", " ").capitalize(),
                                )
                            ),
                        )
                        result["workflow_type"] = "agent2"
                    progress.update(
                        label=(
                            "Matches ready"
                            if result.get("ranked_jobs")
                            else "Search finished"
                        ),
                        state=(
                            "error"
                            if result.get("error")
                            or result.get("status") == "incomplete"
                            else "complete"
                        ),
                        expanded=False,
                    )
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
    react_trace_panel(result)
    if (
        result.get("workflow_type") == "agent3"
        and result.get("skill_gap_analysis")
    ):
        gaps = result["skill_gap_analysis"]
        with st.expander("Recurring skill gaps"):
            recurring = gaps.get("recurring_missing_skills", [])
            if recurring:
                for item in recurring:
                    st.write(f"• {item['skill']} · missing from {item['jobs']} jobs")
            else:
                st.write("No missing skill recurred across multiple ranked jobs.")
            st.caption(gaps.get("recommendation", ""))
    if info.get("skipped_jobs"):
        with st.expander("Why some postings were skipped"):
            for job in info["skipped_jobs"]:
                st.write(f"{job.get('title', 'Posting')}: {job['reason']}")
    if not jobs:
        st.info("No matches were found. Try a broader role or location. Previously tracked jobs are excluded.")
        return
    for index, job in enumerate(jobs, 1):
        match_card(job, index)
    st.divider()
    delivery_panel(result)
    if result.get("status") != "sample":
        st.divider()
        st.subheader("Keep your next step in sight")
        st.write("Your matches are saved. Track an application, add a follow-up note, or prepare for an interview.")
        if st.button("Open my applications →"):
            navigate("Applications")

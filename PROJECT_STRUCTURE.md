# Project structure

The project keeps three agent architectures while sharing search, parsing,
matching, generation, and delivery implementations where their contracts are
compatible.

```text
stage/
├── agent/
│   ├── agent1.py                 ReAct Himalayas workflow
│   ├── agent2.py                 LangGraph LinkedIn workflow and public API
│   ├── agent2_interview.py       Separate on-demand LangGraph: interview PDF
│   ├── agent3.py                 Preserved pre-LangGraph tool-calling workflow
│   ├── agent_core.py             Agent 1 ReAct construction and base tools
│   ├── react_output_parser.py    Agent 1 output validation
│   └── tools/                    LangChain tools used by Agents 1 and 3
│       ├── linkedin_match_tool.py  Agent 3 matching + deterministic tracker save
│       └── interview_preparation.py Agent 3 interview tool (user-triggered)
├── core/
│   ├── agent2_document_extractor.py  Docling + PyPDF extraction
│   ├── agent2_cv_parser.py           Agent 2 CV parsing
│   ├── agent2_job_parser.py          Agent 2 LinkedIn-job parsing
│   ├── agent2_parser_common.py       Shared Agent 2 parser validation
│   ├── agent2_parser.py              Stable parser import facade
│   ├── cosine_matcher.py             Skill, experience, and education scoring
│   ├── esco_normalizer.py            ESCO concept normalization
│   ├── extraction_cache.py           Persistent content-addressed cache
│   ├── cv_parser.py                   Agent 1 CV parser and shared CV model
│   ├── job_parser.py                  Agent 1 parser and shared job model
│   └── matcher.py                     Agent 1 compatibility calculation
├── pipeline/
│   ├── linkedin_cosine_pipeline.py    Agent 2 search-to-ranking pipeline
│   ├── cover_letter.py                Agent 2 cover-letter generation
│   ├── send_results_email.py          Gmail draft creation
│   └── send_results_telegram.py       Telegram delivery
├── search/
│   ├── job_search.py              Himalayas search
│   └── job_scraper.py             LinkedIn browser scraper
├── storage/
│   ├── agent2_database.py         Dedicated Agent 2 SQLite schema/connections
│   └── agent2_checkpointer.py     Persistent Agent 2 LangGraph pause/resume
├── services/
│   ├── application_tracker.py     Shared application tracker (Agents 2 and 3)
│   └── interview_preparation.py   Shared interview generation, PDF, persistence
├── data/
│   └── skills_en.csv              Local ESCO skills vocabulary
├── runtime/                        Ignored Agent 2 runtime databases
├── fixtures/                      Reviewed parser inputs and outputs
├── test/                          Agent, parser, scraper, and matcher tests
├── dev/                           Fixture capture, calibration, and benchmarks
├── config.py                      Model/provider configuration
└── requirements.txt               Runtime dependencies
```

## Agent 2 execution path

```text
test/test_agent2.py or future Streamlit UI
    -> agent/agent2.py
    -> core/agent2_document_extractor.py
    -> core/agent2_cv_parser.py
    -> pipeline/linkedin_cosine_pipeline.py
       -> search/job_scraper.py
       -> core/agent2_job_parser.py
       -> core/esco_normalizer.py
       -> core/cosine_matcher.py
    -> pipeline/cover_letter.py
    -> user approval interrupt
    -> pipeline/send_results_email.py or send_results_telegram.py
```

Only `agent.agent2.run_agent2_full_auto`,
`agent.agent2.run_agent2_full_auto_from_pdf`, and
`agent.agent2.resume_agent2_workflow` are public Agent 2 search entry points.

## Interview preparation (user-triggered, never automatic)

Interview packs cost thousands of LLM tokens, so neither agent generates them
during a job search. They are produced only when a user selects one saved
application and explicitly asks (a Streamlit button later). Both agents call
the same service steps; only the orchestration differs.

```text
services/interview_preparation.py            shared implementation
    generate_interview_content()             prompt + Groq structured output
    render_interview_preparation_pdf()       ReportLab PDF
    persist_interview_preparation()          interview_preparations row
    generate_interview_preparation()         compatibility wrapper (all three)

agent/agent2_interview.py                    Agent 2: separate LangGraph
    START -> load_application -> validate_application
          -> generate_interview_content -> render_and_persist_pdf
          -> finalize -> END
    run_agent2_interview_preparation(application_id, workflow_id=None)
    checkpointed by storage/agent2_checkpointer.py (own thread id)

agent/tools/interview_preparation.py         Agent 3: one ReAct tool
    generate_interview_preparation_pdf(application_id)
    agent.agent3.run_agent3_interview_preparation(application_id)
```

The candidate profile is read from `candidate_profiles` in SQLite, so no CV is
parsed again.

## Shared persistence

Agents 2 and 3 write to the same `runtime/agent2.sqlite3` through
`services.application_tracker.save_application()`, so their recommendations
can be compared fairly. Agent 2 does this in its `persist_recommendations`
node; Agent 3 does it deterministically inside `match_linkedin_jobs_for_agent`
right after ranking (no separate LLM tool). A persistence failure becomes a
`persistence_warning` and never discards a valid ranking. Agent 1 keeps its
independent `cache/seen_jobs_memory.db`.

## Agent 3 tools

```text
match_linkedin_jobs_for_agent     automatic  scrape, parse, score, rank, save
write_cover_letter                automatic  top-ranked job only
ask_user_delivery_channel         automatic  Gmail or Telegram
send_results_draft                automatic  exactly one delivery tool
send_results_telegram             automatic  exactly one delivery tool
generate_interview_preparation_pdf on request  never during the search run
```

## Offline tests

```text
python test/test_agent2_database.py     schema, tracker, interview PDF service
python test/test_agent2.py --offline    search graph + interview graph
python test/test_agent3.py --offline    tool-calling loop, persistence, tool
```

The files under `dev/` are development utilities, not production workflow
dependencies. Files under `test/` are import-safe: live demonstrations run only
when their files are executed directly.

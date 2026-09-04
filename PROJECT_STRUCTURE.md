# Project structure

The project keeps three agent architectures while sharing search, parsing,
matching, generation, and delivery implementations where their contracts are
compatible.

```text
stage/
├── agent/
│   ├── agent1.py                 ReAct Himalayas workflow
│   ├── agent2.py                 LangGraph LinkedIn workflow and public API
│   ├── agent3.py                 Preserved pre-LangGraph tool-calling workflow
│   ├── agent_core.py             Agent 1 ReAct construction and base tools
│   ├── react_output_parser.py    Agent 1 output validation
│   └── tools/                    LangChain tools used by Agents 1 and 3
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
│   └── application_tracker.py     Agent 2 application business rules
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
`agent.agent2.resume_agent2_workflow` are public Agent 2 entry points.

The files under `dev/` are development utilities, not production workflow
dependencies. Files under `test/` are import-safe: live demonstrations run only
when their files are executed directly.

# Project structure

Next Chapter is a Python 3.12 package with a local Streamlit launcher. The
three agent variants share parsing, matching, generation, and persistence.

```text
stage/
├── app.py                         Streamlit launcher
├── pyproject.toml                 Package metadata and build settings
├── requirements*.txt              Existing runtime, lock, and CI dependencies
├── README.md                      Setup, commands, and data handling
├── .streamlit/config.toml          Local server and theme
├── .github/workflows/checks.yml    Offline CI and wheel build
├── src/next_chapter/
│   ├── config.py                  Role-specific model factories
│   ├── paths.py                   Workspace data locations
│   ├── ui/                        Navigation, profile, matches, applications
│   │   ├── components.py          Shared rendering and delivery controls
│   │   ├── support.py             CV uploads and sample data
│   │   └── assets/workspace.css   Packaged interface styles
│   ├── agents/                    Agent 1, 2, 3 and the interview graph
│   │   ├── agent1_support.py      Agent 1 ReAct foundation
│   │   └── tools/                 Agent-facing adapters
│   ├── parsing/                   PDF text extraction and parser schemas
│   │   ├── agent2_document_extractor.py  Lightweight PyPDF extraction
│   │   ├── agent2_cv_parser.py    CV prompt and extraction orchestration
│   │   ├── agent2_cv_rules.py     Deterministic CV cleanup and metadata rules
│   │   ├── agent2_job_parser.py   Job prompt and extraction orchestration
│   │   └── agent2_job_rules.py    Deterministic job section and skill rules
│   ├── matching/                  Cosine, LLM, ESCO and shared scoring
│   ├── search/                    himalayas.py and linkedin.py
│   ├── pipelines/                 LinkedIn search-to-ranking coordination
│   ├── services/                  Tracker, letters, interviews, delivery journal
│   ├── delivery/                  Gmail draft and Telegram implementations
│   └── storage/                   SQLite, checkpoints, extraction cache, memory
├── tests/                         Offline regression suites
│   ├── support/                   Shared workflow fakes
│   └── fixtures/                  parsers/ and esco/
├── examples/                      Explicitly invoked live demonstrations
├── scripts/                       Capture, inspect, provider probes, checks
├── evaluation/                    Replay, benchmarks, calibration and labels
├── docs/                          Architecture and historical evaluation
└── Local, ignored files:
    .env, credentials.json, token.json, .venv/,
    cv/, data/, cache/, runtime/, output/, tmp/
```

All production imports begin with `next_chapter`. Install it once with
`python -m pip install --no-deps --no-build-isolation -e .` after installing
the requirements. The root `app.py` retains the `streamlit run app.py` command.

## Location changes

| Previous location | Current location |
|---|---|
| `agent/` | `src/next_chapter/agents/` |
| `core/agent2_*`, `core/cv_parser.py`, `core/job_parser.py` | `src/next_chapter/parsing/` |
| `core/*matcher*`, `core/esco_normalizer.py` | `src/next_chapter/matching/` |
| `core/extraction_cache.py`, `core/seen_jobs_memory.py` | `src/next_chapter/storage/` |
| `pipeline/linkedin_cosine_pipeline.py` | `src/next_chapter/pipelines/linkedin_matching.py` |
| `pipeline/cover_letter.py` | `src/next_chapter/services/cover_letter.py` |
| `pipeline/send_results_email.py`, `pipeline/send_results_telegram.py` | `src/next_chapter/delivery/gmail.py`, `telegram.py` |
| `search/job_search.py`, `search/job_scraper.py` | `src/next_chapter/search/himalayas.py`, `linkedin.py` |
| `services/ui_support.py` | `src/next_chapter/ui/support.py` |
| `test/` | `tests/` for checks; `examples/` for live demos |
| `fixtures/` | `tests/fixtures/parsers/` |
| `dev/` | `scripts/` and `evaluation/` |

Local databases, credentials, and generated-file locations retain their
existing names. The extraction cache version changed with the PyPDF backend.
Agent numbers remain meaningful for comparison; shared implementations have
not been deleted as legacy code.

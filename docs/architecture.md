# Next Chapter architecture

Next Chapter is a single-user Streamlit workspace. Python runs in the selected
deployment environment; configured model providers and delivery services are
external. PyPDF extraction, cached MiniLM embeddings, cosine scoring, SQLite
storage, and PDF rendering execute in that environment.

## Workflows

```text
app.py → next_chapter.ui.app
  profile → dual-view PyPDF extraction → structured CV parsing → review
  matches → workflow choice
              Agent 2 LangGraph
                query → LinkedIn search → parse jobs → cosine ranking
                → save applications → cover letter → checkpointed approval pause
              Agent 3 classic ReAct
                search action → evaluate action → skill-gap action
                → cover-letter action → visible action/observation trace
            → reviewed delivery → Gmail draft or Telegram message
  applications → status/notes/downloads
                 → separate Agent 2 interview graph → saved PDF
```

Agent 1 uses the original ReAct Himalayas workflow. Streamlit exposes both
Agent 2's explicit LinkedIn LangGraph and Agent 3's classic LinkedIn ReAct
workflow. Agent 3's model emits textual Thought/Action/Action Input steps and
reads each real Observation before continuing. Separate run-scoped tools expose
LinkedIn scraping, deterministic parsing/ranking, recurring skill-gap analysis,
and cover-letter generation. It uses the same algorithms and application
services as Agent 2. Delivery is selected outside the ReAct loop, and Agent 3's
separate interview tool invokes the shared preparation service on request.

## Responsibilities and dependencies

- `ui` owns Streamlit widgets and session navigation.
- `agents` owns workflow control and agent tool adapters.
- `pipelines` coordinates LinkedIn retrieval, parsing, and ranking.
- `parsing` owns schemas, document extraction, prompts, and deterministic rules.
  Each Agent 2 parser keeps its public schema and extraction orchestration in a
  `*_parser.py` module, while pure cleanup and inference rules live in the
  matching `*_rules.py` module.
- `matching` owns compatibility calculations and ESCO normalization.
- `search` owns Himalayas HTTP access and LinkedIn browser scraping.
- `services` owns application tracking, cover letters, interview preparation,
  and delivery receipts.
- `delivery` owns Gmail draft and Telegram integration.
- `storage` owns database connections, LangGraph checkpoints, parsing cache,
  and Agent 1's seen-job memory.
- `config` selects providers/models; `paths` resolves local workspace files.

The older CV/job schemas and experience/education helpers are still shared by
the current parsers and cosine matcher. Both parser families remain available.
The parser facade re-exports the Agent 2 parsers inside the package.

## Persistence and restart recovery

Agents 2 and 3 share `runtime/agent2.sqlite3`; Agent 1 uses
`cache/seen_jobs_memory.db`. No schema change accompanies this reorganization.
The graph stores application state as dictionaries and recreates profile
models at its public interface. Workflow node names and IDs are preserved.
Agent 2 search IDs in the Streamlit URL restore paused graph checkpoints across
restarts. Agent 3's trace and pending delivery are session state, while its
ranked applications remain in the shared database.

`cache/` keeps content-addressed extraction and parser results. The PyPDF
extractor stores readable and layout-oriented views under its own cache version,
so results from former extractors are not reused. Interview PDFs stay in
`output/` and their saved paths remain valid.
Credentials and `.env` stay in the workspace root.

## Development and verification

Install the package in editable mode and run `python -m scripts.run_checks`.
The runner uses fake external services, temporary databases, and a network
guard. Live demos are under `examples/`; they are not automated tests.
`evaluation/` retains the reviewed inputs, provisional labels, and calibration
workflow. `scripts/` holds capture and inspection entry points.

Build a wheel with `python -m pip wheel --no-deps --no-build-isolation
--wheel-dir output/wheels .`. UI CSS is included as package data; private
workspace files, fixtures, and development tools are not packaged.

## Work deferred from this cleanup

Module-level state in Agent 1 and the shared cover/delivery adapters, Telegram's
reuse of email formatting, and mixed generation/rendering responsibilities in
the interview service remain follow-up refactors. Matching rules, model choices,
and delivery behavior are preserved.

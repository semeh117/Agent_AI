# Next Chapter — AI job-search workspace

A local Streamlit app for reviewing a CV, finding and explaining job matches,
editing a cover letter, tracking applications, and preparing for interviews.
Agent 2 orchestrates the workflow with LangGraph; Agents 1 and 3 remain available
for comparison through `next_chapter.agents` and the live demos in `examples/`.

## Start the app

Use Python **3.12**. The checked-in dependency snapshot was captured on Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Copy `.env.example` only on first setup; do not overwrite an existing `.env`.
Open http://127.0.0.1:8501. The server binds to localhost by default.
The **Explore a sample workspace** button uses fictional jobs and illustrative
scores; it makes no model calls and sends no messages.

`requirements.txt` lists direct runtime dependencies; `requirements-lock.txt`
pins the installed environment. Recreate and validate a clean environment before
updating the lock snapshot. Linux/macOS users should create a Python 3.12 venv
and use `python -m pip install -r requirements-lock.txt` followed by
`python -m pip install --no-deps --no-build-isolation -e .`; that platform setup
has not been verified here.

The editable installation connects `next_chapter` to `src/next_chapter/`, so
source edits take effect without reinstalling. Run commands from the project
root. `app.py` is the small launcher; page code and CSS live under
`src/next_chapter/ui/`. See [project structure](docs/project_structure.md)
and [architecture](docs/architecture.md).

## Configure a live run

Edit `.env` using the role-specific provider and model settings in the example.
With its defaults, CV/job parsing uses `OPENROUTER_API_KEY`, orchestration and
interview preparation use `GROQ_API_KEY`, and letters use `GEMINI_API_KEY`.
API usage is charged by your configured providers. Query generation falls back
to a deterministic query if its model fails; a user-entered query skips that call.

Install Google Chrome for local searches. Selenium detects a standard Chrome
installation automatically; set `CHROME_BINARY` and `CHROMEDRIVER_PATH` only for
nonstandard executable locations. Streamlit Community Cloud installs Chromium
and its matching driver from `packages.txt`. Scraping requires internet access
and can fail if LinkedIn serves a login page, restriction, or changed markup.

The first semantic match checks the local Hugging Face cache and downloads the
public MiniLM model if it is missing. The loaded model is reused for later
searches in the same app process. Keep `EMBEDDING_LOCAL_ONLY=false` on fresh
Streamlit Cloud deployments. An optional `HF_TOKEN` increases Hub download rate
limits. Set local-only mode to `true` only after the model is cached and when
strict offline operation is required.

Place the English ESCO skills CSV at `data/skills_en.csv`, or set
`ESCO_SKILLS_PATH`. Obtain the vocabulary from the European Commission's ESCO
download service. Expected columns are `conceptUri`, `preferredLabel`, and
`altLabels`. Without it, the matcher continues with literal, alias, capability,
and semantic matching; ESCO normalization is unavailable.

## Use the workspace

1. Upload a text-based PDF of up to 10 MB. Scanned or image-only PDFs are not
   supported; export the CV directly from Word, Google Docs, or Canva so it has
   selectable text. Review and correct the extracted profile, then select
   **Save reviewed profile**.
2. Open **Job matches**. Choose a query, location, result count and search pool.
   All valid jobs in the selected pool are scored before the top results are
   selected. Small pools limit cost and runtime; a larger pool allows more comparison.
   The effective pool is at least the requested result count. Searches cover
   the last 30 days and exclude jobs already tracked for the candidate.
3. Review matched/missing requirements and edit the top-job cover letter.
   Download it or explicitly approve a delivery channel.
4. Open **Applications** to update status, append notes, reuse a saved profile,
   download letters, or request an interview PDF for a selected application.

Scores are compatibility heuristics, not hiring probabilities. Missing experience
or education requirements incur no penalty. Jobs without extractable skills are
labelled inconclusive. Search/parser errors remain visible alongside valid results.

## Delivery and restart recovery

Gmail: place a Desktop OAuth client file at `credentials.json`. The first delivery
opens Google's consent flow and stores `token.json`. The app creates a draft
addressed to the candidate's email; it does not send an application email.
The `gmail.compose` OAuth scope also permits sending, although this code only creates drafts.

Telegram: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; message the bot first.
Approval sends messages directly to that configured chat. Successful message
parts are persisted, so retrying a known rejection skips already completed parts.
After a timeout, server error, crash, or unreadable response, the outcome may be
uncertain. The app asks you to check the chat and record whether that part arrived
before retrying. It cannot guarantee exactly-once delivery across an external API.

Search IDs are retained in the page URL. Keep the URL or copy the ID from the
sidebar to restore the checkpoint after restarting the app. Approval is accepted
only for a paused workflow; failed delivery has a separate retry action that
preserves the approved content and does not repeat scraping or generation.

## Verification

```powershell
.\.venv\Scripts\python.exe -m scripts.run_checks
.\.venv\Scripts\python.exe -m evaluation.calibrate_cosine_threshold --check
.\.venv\Scripts\python.exe -m evaluation.calibrate_cosine_threshold --output output/evaluation.json
.\.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation --wheel-dir output/wheels .
.\.venv\Scripts\python.exe -m scripts.check_package output/wheels/next_chapter-0.1.0-py3-none-any.whl
```

The first command runs workflow, SQLite/PDF, parser, matcher, delivery, and
Streamlit interaction tests with fake external services and temporary databases.
It blocks outbound connections and saves detailed logs under `output/checks`.
CI runs the same checks on Windows/Python 3.12 using a smaller test dependency set.

The evaluation uses the current parser replay, job IDs, and a hashed fixture
manifest. It includes one captured CV/five jobs and five synthetic profiles from
different technical domains. Labels are **assistant-authored and provisional**:
independent domain review and additional real CVs are still needed. Results are
reported separately for captured and synthetic inputs, with profile-held-out
threshold selection. ESCO is disabled in this evaluation to isolate skill rules
and embeddings. The script does not change the production threshold.

[The embedding comparison](docs/embedding_model_comparison.md) preserves the
historical results; its numbers must not be treated as current accuracy.
Generate the current report with the evaluation command above. Its labels are
provisional and are not an independent accuracy assessment.

## Live demonstrations and development utilities

Automated checks live in `tests/`; executing them does not start a live demo.
Live examples are separate and can call providers, scrape jobs, generate
documents, or deliver results when their workflow requests it:

```powershell
.\.venv\Scripts\python.exe -m examples.run_agent1
.\.venv\Scripts\python.exe -m examples.run_agent2
.\.venv\Scripts\python.exe -m examples.run_agent3
.\.venv\Scripts\python.exe -m examples.application_tracker --show-live
```

`scripts/` contains inspection, fixture capture, model checks, and the offline
runner. `evaluation/` contains replay, calibration, and live model benchmarks.
Reviewed inputs live in `tests/fixtures/parsers/`; benchmark output goes to
`output/benchmarks/`. Run these utilities with `python -m scripts.<name>` or
`python -m evaluation.<name>` from the project root.

## Local data

`runtime/agent2.sqlite3` stores profiles, applications, workflow checkpoints and
delivery receipts. `cache/` stores extracted/parsed content; `output/` contains
generated files. PyPDF extracts selectable CV text in the app environment; no
OCR model is downloaded or run. Original temporary upload files are removed
after extraction. CV text goes to the configured providers during parsing and
generation.

The app is intended for a trusted single-user local workspace: it has no account
login or per-user access controls. Keep localhost binding for this version.
Do not commit `.env`, OAuth files, `.streamlit/secrets.toml`, CVs, runtime databases,
or generated private documents. Back up `runtime/` and `output/` together if you
want to preserve history and interview downloads.

`next_chapter.paths` resolves these locations from the editable checkout.
For an installed wheel, set `NEXT_CHAPTER_HOME` to your workspace directory
or launch from that directory. Local data stays outside the installed package.

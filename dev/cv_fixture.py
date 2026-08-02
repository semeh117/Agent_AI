"""
cv_fixture.py
--------------
Deterministic, offline stand-in for extract_cv_info() — reads a frozen
CVInfo snapshot from disk instead of calling the LLM.

Separate from core/cv_parser.py on purpose, same reasoning as
search/job_search_fixture.py: the real extraction path is what the
product actually uses; this is only for tests that need a candidate
profile held constant across runs.

"""

import json
from pathlib import Path
from core.cv_parser import CVInfo

DEFAULT_FIXTURE_PATH = "fixtures/cv_info_fixture.json"


def load_cv_fixture(fixture_path: str = DEFAULT_FIXTURE_PATH) -> CVInfo:
    """
    Returns a real CVInfo object reconstructed from a frozen fixture file
    (produced by dev/capture_cv_fixture.py), instead of calling the LLM.

    Raises FileNotFoundError with a clear message if the fixture hasn't
    been captured yet — deliberately not swallowed, since a missing CV
    fixture should stop a test run rather than silently returning None.
    """
    path = Path(fixture_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CV fixture not found at {fixture_path}. "
            f"Run: python -m dev.capture_cv_fixture <path_to_cv.pdf>"
        )

    fixture = json.loads(path.read_text(encoding="utf-8"))
    cv_info = CVInfo(**fixture["cv_info"])

    print(f"Loaded CVInfo from fixture '{fixture_path}' "
          f"(source: {fixture.get('source_cv_path', 'unknown')}) "
          f"-> {cv_info.full_name}, {len(cv_info.skills)} skills")

    return cv_info
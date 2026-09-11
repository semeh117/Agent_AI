"""Workspace data locations, independent of individual module locations.

An editable checkout defaults to its repository root. For an installed wheel,
set NEXT_CHAPTER_HOME to the workspace, or launch from that workspace directory.
Existing cache/database/OAuth/output locations are intentionally preserved.
"""
from pathlib import Path
import os


def _workspace_root() -> Path:
    configured = os.environ.get("NEXT_CHAPTER_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src/next_chapter").is_dir():
        return checkout
    return Path.cwd().resolve()


PROJECT_ROOT = _workspace_root()
CACHE_DIR = PROJECT_ROOT / "cache"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
CV_DIR = PROJECT_ROOT / "cv"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"

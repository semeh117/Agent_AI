"""Content-addressed cache for expensive extraction and parsing calls.

The cache is used for CV document extraction, CV parsing, and job requirement
parsing. It is keyed by exact input content rather than a filename or title.
This means:

- The same CV file parsed twice (even across separate runs) reuses the
  cached result instead of re-calling the LLM.
- The same job posting seen twice (e.g. it reappears in a later search)
  reuses its cached requirements too.
- If the underlying text changes even slightly (edited CV, updated
  posting), the hash changes and a fresh extraction happens
  automatically — no manual invalidation needed, ever.

This is NOT "agent memory" in the broader sense (preferences, session
history, learned skill-equivalences) — it's a narrower, purely
mechanical optimization: don't pay for (and wait on) the same LLM call
twice for byte-identical input. Kept deliberately separate from
whatever session/preference memory gets built later, so the two don't
get tangled together.
"""

import hashlib
import json
import os
from pathlib import Path
from next_chapter.paths import PROJECT_ROOT
import tempfile
from typing import Optional, TypeVar
from pydantic import BaseModel

CACHE_DIR = PROJECT_ROOT / "cache"

T = TypeVar("T", bound=BaseModel)


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(namespace: str, key: str) -> Path:
    return CACHE_DIR / namespace / f"{key}.json"


def get_cached(namespace: str, text: str, model_cls: type[T]) -> Optional[T]:
    """
    Returns a cached, already-parsed Pydantic object if this EXACT text
    was extracted before under this namespace. None on a cache miss —
    the caller then does the real LLM call as normal.
    """
    key = _hash_key(text)
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return model_cls(**data)
    except Exception:
        # Corrupted or unreadable cache entry — treat as a miss rather
        # than crash the whole extraction; a fresh LLM call will
        # overwrite this entry with a valid one right after.
        return None


def set_cached(namespace: str, text: str, result: BaseModel) -> None:
    key = _hash_key(text)
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{key}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(result.model_dump_json(indent=2))
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

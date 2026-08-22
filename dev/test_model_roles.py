"""Small live smoke test for Agent 2's three configurable model roles.

This intentionally sends tiny prompts. It verifies provider credentials,
model identifiers, structured extraction, and ordinary text generation before
running the much more expensive LinkedIn workflow.
"""

from __future__ import annotations

import os
import time

from pydantic import BaseModel, Field

from config import get_agent_llm, get_cover_letter_llm, get_parser_llm


class ExtractionProbe(BaseModel):
    skills: list[str] = Field(description="Technical skills present in the text")
    experience_years: float = Field(description="Explicit years of experience")


def _content_text(response) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _run(name: str, call) -> bool:
    started = time.perf_counter()
    try:
        output = call()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"[FAIL] {name} ({elapsed:.2f}s): {type(exc).__name__}: {exc}")
        return False

    elapsed = time.perf_counter() - started
    print(f"[PASS] {name} ({elapsed:.2f}s): {output}")
    return True


def main() -> int:
    print("Configured Agent 2 model roles:")
    print(
        "  parser:",
        os.getenv("PARSER_PROVIDER", "openrouter"),
        os.getenv("PARSER_MODEL", "qwen/qwen-2.5-7b-instruct"),
    )
    print(
        "  agent:",
        os.getenv("AGENT_PROVIDER", "groq"),
        os.getenv("AGENT_MODEL", "openai/gpt-oss-120b"),
    )
    print(
        "  cover:",
        os.getenv("COVER_LETTER_PROVIDER", "gemini"),
        os.getenv("COVER_LETTER_MODEL", "gemini-2.5-flash-lite"),
    )

    results = [
        _run(
            "parser structured output",
            lambda: get_parser_llm()
            .with_structured_output(ExtractionProbe)
            .invoke("Candidate explicitly has Python and SQL and 2 years of experience."),
        ),
        _run(
            "agent text generation",
            lambda: _content_text(
                get_agent_llm().invoke(
                    "Reply with exactly AGENT_OK and no other text."
                )
            ),
        ),
        _run(
            "cover-letter text generation",
            lambda: _content_text(
                get_cover_letter_llm().invoke(
                    "Write one professional sentence stating that Sam is applying "
                    "for a Python Engineer role. Do not invent other details."
                )
            ),
        ),
    ]

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

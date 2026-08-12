"""
Centralized configuration. Swap LLM provider via .env without touching
any other file.

Two independent model "roles" are configurable separately:
- EXTRACTION: used for CV parsing, job parsing, and skill matching —
  tasks needing accurate, grounded, structured output.
- GENERATION: used for cover letter writing — a creative-prose task
  where a different model may be the better fit even if it "lost" the
  extraction comparison.

This lets you run test_model_comparison_kimi_qwen.py, pick whichever
model wins on extraction accuracy, and freely assign the OTHER model to
generation without any code changes — just .env values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _build_llm(provider: str, model: str, temperature: float, max_tokens: int):
    from langchain_openai import ChatOpenAI

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature,
                           api_key=os.getenv("OPENAI_API_KEY"), max_tokens=max_tokens)

    elif provider == "openrouter":
        # OpenRouter exposes an OpenAI-compatible API — both Qwen and
        # Kimi are reachable through the SAME base_url/key, just
        # different model strings. Verify exact model strings at
        # https://openrouter.ai/models before running — Kimi's listing
        # may be under a name like "moonshotai/kimi-k2", but confirm
        # the exact current string rather than trusting this comment.
        return ChatOpenAI(model=model, temperature=temperature,
                           api_key=os.getenv("OPENROUTER_API_KEY"),
                           base_url="https://openrouter.ai/api/v1",
                           max_tokens=max_tokens, max_retries=3)

    elif provider == "groq":
        return ChatOpenAI(model=model, temperature=temperature,
                           api_key=os.getenv("GROQ_API_KEY"),
                           base_url="https://api.groq.com/openai/v1",
                           max_tokens=max_tokens)

    elif provider == "moonshot":
        return ChatOpenAI(model=model, temperature=temperature,
                           api_key=os.getenv("MOONSHOTAI_API_KEY"),
                            base_url=os.getenv("MOONSHOTAI_BASE_URL", "https://api.moonshot.ai/v1"),
                            max_tokens=max_tokens,
                            max_retries=3)
                     
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_extraction_llm(temperature: float = 0.0):
    """CV parsing, job parsing, skill matching — accuracy/grounding matters most."""
    provider = os.getenv("EXTRACTION_LLM_PROVIDER", "openrouter").lower()
    model = os.getenv("EXTRACTION_LLM_MODEL", "moonshot/kimi-k2.6")
    max_tokens = int(os.getenv("EXTRACTION_MAX_TOKENS", "8192"))
    return _build_llm(provider, model, temperature, max_tokens)


def get_generation_llm(temperature: float = 0.3):
    """Cover letter writing — creative prose quality matters most."""
    provider = os.getenv("GENERATION_LLM_PROVIDER", "openrouter").lower()
    model = os.getenv("GENERATION_LLM_MODEL", "qwen/qwen-2.5-7b-instruct")
    max_tokens = int(os.getenv("GENERATION_MAX_TOKENS", "8192"))
    return _build_llm(provider, model, temperature, max_tokens)


# Backward-compat shim — existing call sites using get_llm() keep working
# during migration, routed to the extraction role by default. Once every
# call site below is updated to call get_extraction_llm()/get_generation_llm()
# explicitly, this can be deleted.
def get_llm(temperature: float = 0.0):
    return get_extraction_llm(temperature)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
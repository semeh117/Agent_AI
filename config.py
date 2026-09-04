"""Centralized factories for chat and embedding models.

``get_llm`` remains for the legacy Agent 1 pipeline. Parser, query/tool
orchestration, cover-letter, and embedding workloads use role-specific
factories so changing one model does not silently change every workflow.
"""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_LOCAL_ONLY = os.getenv("EMBEDDING_LOCAL_ONLY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _build_llm(
    provider: str,
    model: str,
    temperature: float,
    role: str | None = None,
):
    """Build one chat model without coupling it to a workflow role."""

    provider = provider.lower()
    role_prefix = f"{role.upper()}_" if role else ""

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            max_tokens=int(
                os.getenv(
                    f"{role_prefix}MAX_TOKENS",
                    os.getenv("OPENROUTER_MAX_TOKENS", "2048"),
                )
            ),
            max_retries=3,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            # The gRPC transport can fail certificate verification on some
            # Windows installations even when ordinary HTTPS providers work.
            # REST uses the environment's normal HTTPS certificate handling
            # and is sufficient for the synchronous calls used here.
            transport="rest",
        )

    if provider == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            # Groq applies its TPM limit to input tokens plus the requested
            # completion budget. A 12k default therefore rejects even small
            # prompts on the 8k on-demand tier before generation starts.
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048")),
            max_retries=3,
            # Groq can close an otherwise successful streamed generation with
            # "Upstream idle timeout exceeded". Bypassing LangChain streaming
            # makes the request retryable by the OpenAI client above and keeps
            # both agents on the same reliable invocation mode.
            disable_streaming=True,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def _model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "openrouter": "qwen/qwen-2.5-7b-instruct",
        "gemini": "gemini-2.5-flash-lite",
        "groq": "qwen/qwen3.6-27b",
        "ollama": "qwen2.5:7b-instruct-q4_K_M",
    }
    if provider not in defaults:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return os.getenv(f"{provider.upper()}_MODEL", defaults[provider])


def get_llm(temperature: float = 0.0):
    """Backwards-compatible Agent 1/global model factory."""

    return _build_llm(
        provider=LLM_PROVIDER,
        model=_model_for_provider(LLM_PROVIDER),
        temperature=temperature,
    )


def get_parser_llm(temperature: float = 0.0):
    """Model used to extract structured information from CVs and jobs."""

    provider = os.getenv("PARSER_PROVIDER", "openrouter").lower()
    model = os.getenv("PARSER_MODEL", "qwen/qwen-2.5-7b-instruct")
    return _build_llm(provider, model, temperature, role="parser")


def get_agent_llm(temperature: float = 0.0):
    """Build the Agent 1/3 orchestrator and Agent 2 query-generation model."""

    provider = os.getenv("AGENT_PROVIDER", "openrouter").lower()
    model = os.getenv(
        "AGENT_MODEL",
        "nvidia/nemotron-3.5-lightning:free",
    )
    return _build_llm(provider, model, temperature, role="agent")


def get_cover_letter_llm(temperature: float = 0.3):
    """Model used only for the final cover-letter generation call."""

    provider = os.getenv("COVER_LETTER_PROVIDER", "gemini").lower()
    model = os.getenv(
        "COVER_LETTER_MODEL",
        "gemini-2.5-flash-lite",
    )
    return _build_llm(provider, model, temperature, role="cover_letter")


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # Agent 2 already has its embedding weights in the Hugging Face
            # cache. Loading cache-only avoids startup metadata requests,
            # optional-file 404s, redirects, and unauthenticated Hub warnings.
            model_kwargs={"local_files_only": EMBEDDING_LOCAL_ONLY},
        )
    except OSError as exc:
        if not EMBEDDING_LOCAL_ONLY:
            raise
        raise RuntimeError(
            f"Embedding model '{EMBEDDING_MODEL}' is not available in the local "
            "Hugging Face cache. Set EMBEDDING_LOCAL_ONLY=false for one run to "
            "download it, then restore EMBEDDING_LOCAL_ONLY=true."
        ) from exc

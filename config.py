"""
Centralized configuration. Swap LLM provider via .env without touching
any other file — all other modules call get_llm() / get_embeddings().
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_llm(temperature: float = 0.0):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        # OpenRouter exposes an OpenAI-compatible API, so we reuse ChatOpenAI
        # and just point it at OpenRouter's base_url with an OpenRouter key.
        #
        # NOTE on the model string: deliberately NOT using ":nitro" here.
        # ":nitro" tells OpenRouter to route to whichever backend provider
        # is fastest — which tends to funnel everyone toward the same one
        # or two "fastest" hosts, making shared-pool rate limits (429s)
        # MORE likely, not less. Plain "qwen/qwen-2.5-7b-instruct" lets
        # OpenRouter load-balance across all available backends instead.
        #
        # NOTE: we tried adding extra_body={"provider": {"ignore": [...]}}
        # to skip a specific rate-limited backend (Phala), but this broke
        # with_structured_output()'s .parse() endpoint — OpenRouter
        # returned choices=None instead of a real completion, likely
        # because the remaining backend(s) after exclusion don't support
        # OpenAI's structured-output/JSON-schema format. Reverted. If a
        # 429 recurs, the safer fix is waiting out retry_after_seconds
        # rather than excluding providers by name.
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct"),
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            max_tokens=8192,
            max_retries=3,
        )

    elif LLM_PROVIDER == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            max_tokens=8492,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
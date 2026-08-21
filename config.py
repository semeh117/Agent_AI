"""
Centralized configuration. Swap LLM provider via .env without touching
any other file — all other modules call get_llm() / get_embeddings().
"""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
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
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct"),
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",

            max_retries=3,
        )
    elif LLM_PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
                    temperature=temperature,
                    google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    elif LLM_PROVIDER == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            # Groq applies its TPM limit to input tokens plus the requested
            # completion budget. A 12k default therefore rejects even small
            # prompts on the 8k on-demand tier before generation starts.
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048")),
            max_retries=3,
        )
    elif LLM_PROVIDER == "ollama":
       from langchain_ollama import ChatOllama
       return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
            temperature=temperature,
    )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

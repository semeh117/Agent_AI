"""
Centralized configuration. Swap LLM provider via .env without touching
any other file — all other modules call get_llm() / get_embeddings().
"""
import os
from dotenv import load_dotenv

load_dotenv() 
 # Load environment variables from .env file
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
    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    elif LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        # OpenRouter exposes an OpenAI-compatible API, so we reuse ChatOpenAI
        # and just point it at OpenRouter's base_url with an OpenRouter key.
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct:nitro"),
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    elif LLM_PROVIDER == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
    )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)



# test_config.py
import time 
from config import get_llm, LLM_PROVIDER

llm = get_llm()
start = time.time()
response = llm.invoke("Say hello in 3 words.")
print(f"LLM_PROVIDER={LLM_PROVIDER}")
print(f"Took {time.time() - start:.2f}s")
print(response.content)

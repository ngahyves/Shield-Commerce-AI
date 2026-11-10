"""LLM invocation with concurrency control and Gemini fallback.

- Semaphore limits concurrent Groq calls (free tier = 8000 TPM).
- On 429, fallback to Gemini (separate quota, 250K+ TPM).
- Gemini calls bypass the Groq semaphore (independent quota).
"""

import os
import threading
from typing import Any

from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

load_dotenv()
log = ProjectLogger(__name__).get_logger()

# Max 2 Groq calls in flight. Free tier TPM = 8000, each /explain ~3600 tokens,
# so 2 concurrent = ~7200 tokens in flight (safe margin below 8000).
_GROQ_SEMAPHORE = threading.Semaphore(2)


def _get_groq():
    from langchain_groq import ChatGroq
    cfg = ConfigLoader()
    return ChatGroq(
        model=cfg.get("agent", "llm_model", default="openai/gpt-oss-20b"),
        temperature=cfg.get("agent", "llm_temperature", default=0.0),
        max_tokens=cfg.get("agent", "llm_max_tokens", default=2000),
        reasoning_effort="low",
    )


def _get_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    cfg = ConfigLoader()
    return ChatGoogleGenerativeAI(
        model=cfg.get("agent", "llm_model_2", default="gemini-3.5-flash-lite"),
        temperature=cfg.get("agent", "llm_temperature", default=0.0),
        max_output_tokens=cfg.get("agent", "llm_max_tokens", default=2000),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        request_timeout=45.0,
    )


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def invoke_with_fallback(
    messages: list[BaseMessage],
    structured_schema: Any = None,
) -> tuple[Any, str]:
    """Invoke Groq under semaphore with timeout. On 429 or slot timeout, use Gemini."""
    acquired = _GROQ_SEMAPHORE.acquire(timeout=10)
    
    if not acquired:
        # Queue is saturated, skip Groq entirely
        log.warning("groq semaphore timeout (queue saturated) — going to gemini")
        llm = _get_gemini()
        if structured_schema is not None:
            llm = llm.with_structured_output(structured_schema)
        return llm.invoke(messages), "gemini"
    
    try:
        llm = _get_groq()
        if structured_schema is not None:
            llm = llm.with_structured_output(structured_schema)
        response = llm.invoke(messages)
        return response, "groq"
    except Exception as exc:
        if not _is_rate_limit(exc):
            raise
        log.warning("groq rate-limited — falling back to gemini")
    finally:
        _GROQ_SEMAPHORE.release()
    
    llm = _get_gemini()
    if structured_schema is not None:
        llm = llm.with_structured_output(structured_schema)
    return llm.invoke(messages), "gemini"
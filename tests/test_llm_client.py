"""Tests for invoke_with_fallback: Groq -> Gemini resilience."""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.llm_client import invoke_with_fallback


MESSAGES = [HumanMessage(content="test")]


def test_returns_groq_on_success():
    with patch("src.agents.llm_client._get_groq") as get_groq:
        get_groq.return_value.invoke.return_value = "groq response"

        response, provider = invoke_with_fallback(MESSAGES)

    assert provider == "groq"
    assert response == "groq response"


def test_falls_back_to_gemini_on_429():
    with patch("src.agents.llm_client._get_groq") as get_groq, \
         patch("src.agents.llm_client._get_gemini") as get_gemini:

        get_groq.return_value.invoke.side_effect = Exception("429 rate limit exceeded")
        get_gemini.return_value.invoke.return_value = "gemini response"

        response, provider = invoke_with_fallback(MESSAGES)

    assert provider == "gemini"
    assert response == "gemini response"


def test_raises_on_non_rate_limit_error():
    with patch("src.agents.llm_client._get_groq") as get_groq:
        get_groq.return_value.invoke.side_effect = ValueError("invalid request")

        with pytest.raises(ValueError, match="invalid request"):
            invoke_with_fallback(MESSAGES)
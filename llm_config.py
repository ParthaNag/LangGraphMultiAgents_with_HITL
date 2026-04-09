"""
llm_config.py — Switchable LLM Provider
-----------------------------------------
Single source of truth for which LLM is in use.
Both agents.py and dashboard.py import from here.

  set_provider("gemini")                          → use Google Gemini API
  set_provider("ollama", "llama3.2:latest")       → use local Ollama
  set_provider("openrouter", "openai/gpt-4o-mini") → use OpenRouter

The _config dict lives in this module's namespace, which Python caches in
sys.modules. So the setting persists across Streamlit reruns without any
session_state tricks.
"""

import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

_config: dict = {
    "provider":          "gemini",           # "gemini" | "ollama" | "openrouter"
    "gemini_model":      "gemini-2.5-flash",
    "ollama_model":      "llama3.2:latest",
    "openrouter_model":  "openai/gpt-4o-mini",
}

# Popular OpenRouter models — mix of free and paid tiers
OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct:free",
    "mistralai/mistral-small",
    "deepseek/deepseek-chat",
]

_cache: dict = {}   # (provider, model, json_mode) → LLM instance


# ── Public API ────────────────────────────────────────────────────────────────

def set_provider(provider: str, model: str | None = None) -> None:
    """Switch the active provider (and optionally the model)."""
    _config["provider"] = provider
    if model:
        if provider == "gemini":
            _config["gemini_model"] = model
        elif provider == "openrouter":
            _config["openrouter_model"] = model
        else:
            _config["ollama_model"] = model
    _cache.clear()   # force fresh instances on next call


def get_provider() -> str:
    return _config["provider"]


def get_active_model() -> str:
    p = _config["provider"]
    if p == "gemini":
        return _config["gemini_model"]
    if p == "openrouter":
        return _config["openrouter_model"]
    return _config["ollama_model"]


def list_openrouter_models() -> list[str]:
    """Return the curated list of OpenRouter models."""
    return OPENROUTER_MODELS


def get_llm():
    """Return the standard (creative) LLM for the active provider."""
    return _get_or_build(json_mode=False)


def get_llm_json():
    """Return a JSON-constrained LLM for structured output nodes."""
    return _get_or_build(json_mode=True)


def list_ollama_models() -> list[str]:
    """Query the local Ollama daemon for available models."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        return models if models else ["llama3.2:latest"]
    except Exception:
        return ["llama3.2:latest"]


def ollama_running() -> bool:
    """Return True if the local Ollama daemon is reachable."""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False


# ── Internal ──────────────────────────────────────────────────────────────────

def _get_or_build(json_mode: bool):
    key = (_config["provider"], get_active_model(), json_mode)
    if key not in _cache:
        _cache[key] = _build(json_mode)
    return _cache[key]


def _build(json_mode: bool):
    if _config["provider"] == "ollama":
        from langchain_ollama import ChatOllama
        model = _config["ollama_model"]
        if json_mode:
            return ChatOllama(model=model, temperature=0.1, format="json")
        return ChatOllama(model=model, temperature=0.7)

    if _config["provider"] == "openrouter":
        from langchain_openai import ChatOpenAI
        model   = _config["openrouter_model"]
        api_key = os.getenv("OPENROUTER_API_KEY")
        kwargs = dict(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1 if json_mode else 0.7,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    # Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    model   = _config["gemini_model"]
    api_key = os.getenv("GOOGLE_AISTUDIO_API_KEY")
    if json_mode:
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key,
            temperature=0.1, response_mime_type="application/json",
        )
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=api_key, temperature=0.7,
    )

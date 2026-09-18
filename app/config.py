"""Environment-driven configuration for the GridWise LLM service.

Centralizes every env var the service reads so app/main.py and
app/interpreter/llm_client.py share one source of truth instead of each
reading os.environ directly with its own defaults.
"""
from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# ── LLM Provider (AgentRouter -- OpenAI-compatible) ──────────────────────
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "agentrouter")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://agentrouter.org/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
LLM_REQUEST_TIMEOUT_SECONDS = _float_env("LLM_REQUEST_TIMEOUT_SECONDS", 12.0)
# Number of attempts against the primary key (including the first) before
# falling back to LLM_FALLBACK_API_KEY.
LLM_MAX_RETRIES = _int_env("LLM_MAX_RETRIES", 1)
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", LLM_MODEL)

# ── Server ────────────────────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = _int_env("PORT", 8000)
# Hard ceiling on the whole /optimize-energy request (LLM + guardrails + LP
# solve together), independent of the per-call LLM timeout above. Keeps the
# service inside the judge's stated 30s per-request limit even if the LLM
# call itself stays within its own timeout but something downstream stalls.
REQUEST_TIMEOUT_SECONDS = _float_env("REQUEST_TIMEOUT_SECONDS", 25.0)

# ── Optimizer ─────────────────────────────────────────────────────────────
NUMERIC_TOLERANCE = _float_env("NUMERIC_TOLERANCE", 0.01)

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

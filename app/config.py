"""Environment-driven configuration for the GridWise LLM service."""
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


# Anthropic model used for operator-note interpretation. Sonnet 5 is the
# default: strong instruction following at low latency/cost, which matters
# because the judge enforces a 30s per-request timeout and scores p95 <= 5s.
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")

# Hard ceiling on the whole /optimize-energy request (judge timeout is 30s).
# Leaves headroom for guardrail validation + LP solve + response building.
REQUEST_TIMEOUT_SECONDS = _float_env("REQUEST_TIMEOUT_SECONDS", 25.0)

# Per-call timeout for the Anthropic API request itself.
LLM_TIMEOUT_SECONDS = _float_env("LLM_TIMEOUT_SECONDS", 18.0)

# Number of attempts (including the first) made against the LLM before
# falling back to a safe no_op interpretation for every note.
LLM_MAX_ATTEMPTS = _int_env("LLM_MAX_ATTEMPTS", 2)

LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 1024)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = _int_env("PORT", 8000)

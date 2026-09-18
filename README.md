# GridWise — LLM-Assisted Operator Directive Interpretation

Smart Campus Energy Optimization Challenge (BUP CSE Fest 2026 Hackathon, Online Preliminary).

This service accepts a 24-hour campus energy scenario plus 1–3 natural-language operator
notes, uses an LLM to interpret the notes into structured directives, validates that
interpretation deterministically, applies it to a linear-programming optimizer, and returns
a valid, cost-minimized 24-hour battery/grid/solar schedule.

## Architecture

```
Energy Data + Operator Notes
        |
        v
  LLM Interpreter        (app/interpreter/llm_client.py, prompts.py)
        |  untrusted structured directives
        v
  Guardrail Validator     (app/interpreter/guardrails.py)
        |  safe, schema-correct directives (unsupported/malformed -> no_op)
        v
  Math Optimizer (LP)     (app/optimizer/model.py, directives.py)
        |  cost-minimal 24h schedule
        v
  Final Validator         (app/optimizer/validator.py)
        |  independent replay against every energy/battery/directive rule
        v
  API Response            (app/main.py, pipeline.py)
```

- **LLM role**: interprets each operator note into one of 6 supported directive types
  (`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`,
  `max_grid_window`, `no_op`) via Anthropic Claude tool-use (structured output). The LLM is
  on the critical path that produces the constraints given to the optimizer — it is not used
  only for `plan_summary`/documentation text.
- **Guardrails**: pure deterministic Python. Rejects unsupported directive types, out-of-range
  or unordered hours, out-of-bounds numeric values, and malformed shapes — falling back to a
  safe `no_op` rather than crashing or inventing a rule. Never trusts LLM output directly.
- **Optimizer**: linear program (PuLP + CBC) that minimizes `sum(grid_kwh[h] * tariff[h])`
  subject to energy balance, battery bounds/rate limits, effective-solar limits, directive
  constraints, and end-of-day battery neutrality.
- **Final validator**: independently replays the produced schedule hour-by-hour against every
  Problem Statement rule before the response is ever returned, catching solver edge cases
  before the judge does.

## Requirements

- Python 3.11+
- An Anthropic API key (for the LLM interpretation step)
- `coinor-cbc` (CBC solver binary; bundled automatically via the `pulp` wheel on most
  platforms, and installed explicitly in the Docker image)

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Auth for the LLM interpretation call |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5` | Model used for operator-note interpretation |
| `LLM_TIMEOUT_SECONDS` | No | `20` | Per-call LLM timeout |

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` locally. Do not commit `.env`.

## Local quickstart (clean environment)

```bash
git clone <this-repo-url>
cd BUP_Hackathon_Preli
git checkout Raisa

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Run a public sample

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  --data-binary @public_samples/sample_grid_101.json
```

Expected shape of the response:

```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    {"note_index": 0, "applies": true,  "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [13, 14], "factor": 0.2}, "explanation": "..."},
    {"note_index": 1, "applies": true,  "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]}, "explanation": "..."},
    {"note_index": 2, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null, "explanation": "..."}
  ],
  "hourly_plan": [ { "hour": 0, "grid_kwh": ..., "solar_used_kwh": ..., "battery_action": "idle", "battery_kwh": 0, "battery_energy_after_kwh": ... }, "... 23 more entries ..." ],
  "total_grid_kwh": 3822.0,
  "total_cost_bdt": 30972.0,
  "peak_grid_kwh": 310.0,
  "plan_summary": "Applied 2 operator directive(s) and produced a cost-minimized 24-hour schedule: ..."
}
```

## Running tests

```bash
python -m pytest tests/ -v
```

29 tests covering: guardrail rejection/normalization of malformed or unsupported LLM output,
per-directive optimizer correctness (each of the 6 directive types individually and combined),
infeasibility handling, energy-balance/battery-neutrality invariants, full API contract
(schema validation, 400 on malformed input, 500 with no leaked internals), and an end-to-end
happy path against the public sample case. All optimizer/guardrail/API tests run fully offline
with no LLM calls (the LLM call itself is isolated behind `app/interpreter/llm_client.py` and
stubbed in tests), so CI does not require an API key.

## Docker

```bash
docker build -t gridwise-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... gridwise-api
curl http://localhost:8000/health
```

The image binds to `0.0.0.0:8000`, contains no baked-in secrets, and takes the API key only
via the `ANTHROPIC_API_KEY` environment variable at run time.

## Model / provider

- Provider: Anthropic (Claude), called via the official `anthropic` Python SDK.
- Model: configurable via `ANTHROPIC_MODEL` (defaults to `claude-sonnet-4-5`).
- The LLM is invoked once per request through tool-use with a strict JSON schema
  (`app/interpreter/prompts.py::TOOL_SCHEMA`), so its output is already structurally
  constrained before the deterministic guardrails run a second, independent check.

## Optimizer / solver

- Linear program built with [PuLP](https://coin-or.github.io/pulp/), solved with the bundled
  CBC solver (`pulp.PULP_CBC_CMD`).
- Decision variables per hour: `grid_kwh`, `solar_used_kwh`, `battery_charge_kwh`,
  `battery_discharge_kwh`, `battery_energy_after_kwh`.
- Objective: `minimize sum(grid_kwh[h] * tariff_bdt_per_kwh[h])`.

## Safe failure behavior

- Malformed JSON / schema violations → `400` with a generic error message.
- Unsolvable/infeasible scenario after valid directives → `422` with a generic error message.
- Any unexpected internal error → `500` with a generic error message; full details are logged
  server-side only, never included in the HTTP response (no stack traces, no secrets).
- LLM provider failure (timeout, auth, rate limit, malformed output) → every operator note for
  that request safely falls back to `no_op` rather than the service crashing or inventing a
  directive.

## Known limitations

- The optimizer assumes all organizer scoring scenarios are feasible with non-contradictory
  directives, as guaranteed by the Problem Statement; a genuinely infeasible input returns
  `422` rather than a best-effort schedule.
- LLM latency is on the request critical path; `LLM_TIMEOUT_SECONDS` bounds it, and a slow or
  unavailable provider degrades to `no_op` for that request's notes rather than failing the
  whole service.
- Grid export/net-metering is out of scope, matching the Problem Statement (unused solar is
  curtailed).

## Credited dependencies

FastAPI, Uvicorn, Pydantic, PuLP (+ CBC), Anthropic Python SDK, pytest, httpx.

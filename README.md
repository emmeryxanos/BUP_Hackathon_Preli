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
  `max_grid_window`, `no_op`) via LLM tool-calling (structured output), against an
  OpenAI-compatible endpoint (AgentRouter, serving Claude models). The LLM is
  on the critical path that produces the constraints given to the optimizer — it is not used
  only for `plan_summary`/documentation text.
- **Guardrails**: pure deterministic Python. Rejects unsupported directive types, out-of-range
  or unordered hours, out-of-bounds numeric values, and malformed shapes — falling back to a
  safe `no_op` rather than crashing or inventing a rule. Never trusts LLM output directly.
- **Optimizer**: linear program (SciPy's HiGHS solver, `scipy.optimize.linprog`) that minimizes
  `sum(grid_kwh[h] * tariff[h])` subject to energy balance, battery bounds/rate limits,
  effective-solar limits, directive constraints, and end-of-day battery neutrality. No external
  solver binary is required (HiGHS is bundled in the `scipy` wheel).
- **Final validator**: independently replays the produced schedule hour-by-hour against every
  Problem Statement rule before the response is ever returned, catching solver edge cases
  before the judge does. If the fully-constrained LP is infeasible, the service returns `422`
  rather than silently relaxing or dropping any directive — see "Known limitations" below.

## Requirements

- Python 3.11+
- An LLM API key for the configured provider (for the LLM interpretation step)
- No external solver binary needed — the optimizer uses SciPy's bundled HiGHS solver.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | No | `agentrouter` | Provider label (informational; the client is OpenAI-compatible) |
| `LLM_API_KEY` | Yes | — | Auth for the primary LLM interpretation call |
| `LLM_MODEL` | No | `claude-haiku-4-5` | Model used for operator-note interpretation |
| `LLM_BASE_URL` | No | `https://agentrouter.org/v1` | OpenAI-compatible base URL |
| `LLM_REQUEST_TIMEOUT_SECONDS` | No | `12` | Per-call LLM timeout |
| `LLM_MAX_RETRIES` | No | `1` | Retries against the primary key/model before falling back |
| `LLM_FALLBACK_API_KEY` | No | — | Backup key, tried once if the primary key/model exhausts its retries |
| `LLM_FALLBACK_MODEL` | No | same as `LLM_MODEL` | Model used for the fallback attempt |
| `PORT` / `HOST` | No | `8000` / `0.0.0.0` | Uvicorn bind address (informational; see quickstart command) |
| `REQUEST_TIMEOUT_SECONDS` | No | `25` | Hard wall-clock budget for the *entire* request (LLM call + guardrails + LP solve), enforced in `app/main.py` via `anyio.move_on_after`; judge's hard limit is 30s |
| `NUMERIC_TOLERANCE` | No | `0.01` | Matches `TOL` in `app/optimizer/validator.py` |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

Copy `.env.example` to `.env` and fill in `LLM_API_KEY` (and optionally `LLM_FALLBACK_API_KEY`)
locally. Do not commit `.env`.

## Local quickstart (clean environment)

```bash
git clone <this-repo-url>
cd BUP_Hackathon_Preli
git checkout Raisa

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set LLM_API_KEY=sk-... (and optionally LLM_FALLBACK_API_KEY=sk-...)

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

124 tests covering: guardrail rejection/normalization of malformed or unsupported LLM output
(including type coercion, NaN/negative/out-of-bounds numeric fields, and note-index handling
in `validate_all`), per-directive optimizer correctness (each of the 6 directive types
individually and combined) plus direct unit tests of the shared directive-effect helpers and
optimizer edge cases (zero demand/tariff hours, solar curtailment, overlapping directives),
every rejection branch of the final replay validator (`app/optimizer/validator.py`) exercised
directly with hand-built invalid schedules, Pydantic request-schema bound checks, the async LLM
client's tool-use call and retry/fallback-key logic with a mocked `httpx.AsyncClient` (no
network/key), the LLM-failure-degrades-to-no_op safety guarantee at both the interpreter and
full-API level, prompt-construction correctness, infeasibility handling end-to-end (422 at the
API layer, not just the optimizer layer), energy-balance/battery-neutrality invariants, full API
contract (schema validation, 400 on malformed input, 404/405 on bad routes, 500 with no leaked
internals), and an end-to-end happy path against the public sample case. All tests run fully
offline with no LLM calls (the LLM call itself is isolated behind
`app/interpreter/llm_client.py` and stubbed/mocked/monkeypatched in tests), so CI does not
require an API key.

**Platform note (Windows only):** a handful of the async `POST /optimize-energy` tests in
`tests/test_api.py` may print `Windows fatal exception: access violation` tracebacks during
teardown on Windows + Python 3.11.x. This is a documented interaction between Starlette
`TestClient`'s sync-to-async thread bridging and the Windows `ProactorEventLoop`; it happens
after each test's own assertions complete and does not change the pass/fail outcome (`pytest`
still exits 0, every test still reports `PASSED`). It was confirmed *not* to reproduce against a
real `uvicorn` server hit with real HTTP requests, so it is a `TestClient`-only artifact on this
OS, not a production risk. See the module docstring in `tests/test_api.py` for detail.

## Docker

```bash
docker build -t gridwise-api .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... gridwise-api
curl http://localhost:8000/health
```

The image binds to `0.0.0.0:8000`, contains no baked-in secrets, and takes the API key only
via the `LLM_API_KEY` environment variable at run time.

## Model / provider

- Provider: AgentRouter, an OpenAI-compatible proxy in front of Claude models, called over
  plain HTTP via `httpx` (no vendor SDK dependency).
- Model: configurable via `LLM_MODEL` (defaults to `claude-haiku-4-5`); a separate
  `LLM_FALLBACK_API_KEY`/`LLM_FALLBACK_MODEL` pair is tried once if the primary key/model
  exhausts `LLM_MAX_RETRIES` attempts.
- The LLM is invoked once per request (per attempt) through OpenAI-style tool-calling with a
  strict JSON schema (`app/interpreter/prompts.py::TOOL_SCHEMA`, converted to OpenAI's
  `tools`/`parameters` shape in `app/interpreter/llm_client.py`), so its output is already
  structurally constrained before the deterministic guardrails run a second, independent check.

## Optimizer / solver

- Linear program built with [SciPy](https://scipy.org/), solved with the bundled HiGHS solver
  (`scipy.optimize.linprog(method="highs")`) — a pure wheel dependency, so no external solver
  binary needs to be installed or discovered at runtime.
- Battery activity per hour is modeled as a single signed variable `delta[h]` (positive =
  charge, negative = discharge) rather than separate non-negative charge/discharge variables,
  which makes "idle" and "never simultaneously charging and discharging" automatic properties
  of the LP rather than constraints that need to be added and policed separately.
- Objective: `minimize sum(grid_kwh[h] * tariff_bdt_per_kwh[h])`.
- If the fully-constrained LP (every validated directive applied) is infeasible, `solve_schedule`
  raises immediately rather than relaxing constraints — see "Known limitations".

## Safe failure behavior

- Malformed JSON / schema violations → `400` with a generic error message.
- Unsolvable/infeasible scenario after valid directives → `422` with a generic error message.
- Any unexpected internal error → `500` with a generic error message; full details are logged
  server-side only, never included in the HTTP response (no stack traces, no secrets).
- LLM provider failure (timeout, auth, rate limit, malformed output) → every operator note for
  that request safely falls back to `no_op` rather than the service crashing or inventing a
  directive. `call_llm_for_directives` never raises: it returns an empty list on total failure,
  which `guardrails.validate_all` pads out to a safe `no_op` per note.
- Whole-request timeout: `app/main.py` wraps the entire pipeline (LLM call + guardrails + LP
  solve) in `anyio.move_on_after(REQUEST_TIMEOUT_SECONDS)`, independent of the per-call LLM
  timeout — a slow LP solve alone can't blow past the judge's stated 30s limit unnoticed. On
  timeout, returns `500` with a generic error message.

## Known limitations

- The optimizer does not silently relax or drop directives on infeasibility, by design: the
  Problem Statement guarantees organizer-valid scoring scenarios are feasible with every
  directive applied (Section 5.1), and explicitly states that "correct extraction without
  correct downstream application does not pass the case" (Section 11.2) — so a fallback that
  drops a directive the judge expects applied would silently fail scoring, whereas a `422` is
  an honest, correctly-scored failure signal. A genuinely infeasible input therefore always
  returns `422` rather than a best-effort schedule with quietly-ignored constraints.
- LLM latency is on the request critical path; `LLM_REQUEST_TIMEOUT_SECONDS` bounds each
  attempt, and a slow or unavailable provider (after retries and the fallback key/model)
  degrades to `no_op` for that request's notes rather than failing the whole service. The
  broader `REQUEST_TIMEOUT_SECONDS` wall-clock budget (see "Safe failure behavior") separately
  bounds the whole request.
- Grid export/net-metering is out of scope, matching the Problem Statement (unused solar is
  curtailed).
- The LP solve itself is synchronous/CPU-bound; `anyio.move_on_after` cancels the *request*
  once its deadline passes, but a solve already in progress on the event loop is not preempted
  mid-computation (true preemption would require running it in a worker thread/process pool,
  which was judged unnecessary given the 24-hour/6-directive problem size solves in
  milliseconds in practice).

## Credited dependencies

FastAPI, Uvicorn, Pydantic, SciPy, NumPy, httpx, AnyIO, pytest, pytest-asyncio.

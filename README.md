# GridWise LLM — Smart Campus Energy Optimization

Solution for the BUP CSE Fest 2026 Hackathon preliminary round ("Smart Campus Energy
Optimization Challenge — LLM-Assisted Operator Directive Interpretation").

Pipeline: **operator notes → LLM interpreter → deterministic guardrails → linear-program
optimizer → final schedule**. The LLM only ever produces *candidate* structured
directives; every field is re-validated by plain Python before it can affect the
optimization, and the optimizer's output is always energy-balance/battery-valid by
construction.

## Architecture

```
Energy data + operator notes
        │
        ▼
 LLM Interpreter (Claude, structured outputs)   app/llm_interpreter.py
        │  candidate directive_interpretation[]  (untrusted)
        ▼
 Guardrail Validator                             app/guardrails.py
        │  exactly one safe, schema-valid entry per note
        ▼
 Directive Application                           app/optimizer.py:apply_directives
        │  effective_solar / no_charge / no_discharge / reserve / grid_cap arrays
        ▼
 LP Optimizer (SciPy HiGHS)                       app/optimizer.py:solve_schedule
        │  cost-minimal, constraint-satisfying 24h plan
        ▼
 API Response                                     app/main.py
```

- **LLM**: Anthropic Claude (`claude-sonnet-5` by default), called through the official
  `anthropic` Python SDK using **Structured Outputs**
  (`output_config.format = json_schema`) so every response is already schema-valid JSON —
  no fragile prompt-only JSON parsing. See `app/llm_interpreter.py`.
- **Guardrails**: pure deterministic Python (`app/guardrails.py`). Rejects unsupported
  directive types, out-of-range numeric values, malformed hour lists, and incomplete
  note coverage — always downgrading a bad entry to `no_op` rather than crashing or
  inventing a rule.
- **Optimizer**: a linear program over 24 hours, solved with `scipy.optimize.linprog`
  (HiGHS method — a pure-library call, no external solver binary, so the Docker image
  stays small and there is no solver-availability risk). Battery activity is modeled as
  one signed variable per hour (`+` = charge, `-` = discharge), which makes "idle" and
  "never charge and discharge simultaneously" automatic rather than something to police
  after the fact.
- **Feasibility fallback**: organizer-valid scenarios are guaranteed feasible with every
  directive applied, but if our own LLM extraction is ever inconsistent, the optimizer
  retries with directive constraints progressively relaxed (reserve → charge/discharge
  windows → grid caps → no directives at all) before falling back to the always-feasible
  plain-GridWise baseline. The service never fails to return a schedule.

## Requirements

- Python 3.12+ (developed and tested on 3.13)
- An Anthropic API key (`ANTHROPIC_API_KEY`)

## Local Quickstart

```bash
git clone <this-repo>
cd gridwise-llm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env and set ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)   # or use a process manager / direnv

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

Sample request (see also `tests/sample_requests/basic.json`):

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @tests/sample_requests/basic.json | python -m json.tool
```

## Environment Variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key used for operator-note interpretation. |
| `LLM_MODEL` | No | `claude-sonnet-5` | Model used for interpretation. |
| `REQUEST_TIMEOUT_SECONDS` | No | `25` | Hard ceiling on total `/optimize-energy` processing (judge timeout is 30s). |
| `LLM_TIMEOUT_SECONDS` | No | `18` | Per-call Anthropic request timeout. |
| `LLM_MAX_ATTEMPTS` | No | `2` | Attempts against the LLM before falling back to safe `no_op` for every note. |
| `LLM_MAX_TOKENS` | No | `1024` | Max output tokens for the interpretation call. |
| `HOST` / `PORT` | No | `0.0.0.0` / `8000` | Bind address for `uvicorn`. |

No secrets are read from anywhere except environment variables; none are committed to
this repository or baked into the Docker image.

## Running the Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite (21 tests) covers:
- `tests/test_guardrails.py` — every guardrail rule in Problem Statement Section 08
  (unsupported types, invalid hours/factor/reserve/grid-cap, missing/duplicate
  note_index, `applies` semantics).
- `tests/test_optimizer.py` — directive application + LP correctness: energy balance
  every hour, end-of-day battery neutrality, per-directive constraint satisfaction
  (`solar_reduction`, `no_charge_window`, `no_discharge_window`,
  `minimum_battery_reserve`, `max_grid_window`), non-negative values, cost
  recomputation.
- `tests/test_api.py` — end-to-end HTTP behavior including the safe-fallback path when
  the LLM call fails (the API's `interpret_notes` dependency is monkeypatched so these
  tests run without a live API key).

## Docker

The image contains no baked-in secrets or `.env` file (see `.dockerignore`); all
configuration is supplied at `docker run` time.

```bash
docker build -t gridwise-llm:local .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... gridwise-llm:local
curl -s http://localhost:8000/health
```

## Known Limitations

- The LLM interpretation call currently supports the Anthropic API only. Swapping to
  another provider means editing `app/llm_interpreter.py`; the rest of the pipeline
  (guardrails, optimizer, API layer) is provider-agnostic.
- If the LLM is unreachable or returns unusable output after `LLM_MAX_ATTEMPTS`
  attempts, every operator note for that request is safely treated as `no_op` rather
  than the service failing — this trades interpretation credit for that one request for
  guaranteed uptime, per the Problem Statement's "SAFE FAILURE" requirement (Section 08).
- The optimizer's feasibility-relaxation fallback (see Architecture above) only
  activates when our own extracted directives make the LP infeasible; it is not
  expected to trigger on organizer-valid ground-truth scenarios, which are guaranteed
  feasible by the Problem Statement.

## Credits / Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — HTTP API layer.
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API client.
- [SciPy](https://scipy.org/) (`optimize.linprog`, HiGHS method) — LP solver.
- [Pydantic](https://docs.pydantic.dev/) — request/response schema validation.

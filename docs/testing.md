# Testing strategy

## Overview

- **22 tests**, 6 files, ~35 s runtime
- **46% global coverage** — ~85% on critical serving modules, 0% on offline jobs
- **Zero external calls**: no LLM tokens, no Redis, no Qdrant, no Gold Parquet in the test path
- Framework: `pytest` + `pytest-cov` + FastAPI `TestClient`

## Philosophy

The project separates two kinds of code, and they get different testing treatment:

| Code type | Location | Testing approach |
|-----------|----------|------------------|
| **Online** (serving) | `app/`, `src/agents/` | Unit + integration tests — **85%+ on critical paths** |
| **Offline** (batch) | `src/training/`, `src/evaluation/`, `src/vector/indexer.py` | Validated in notebook — **0% unit coverage** |

This is intentional. Offline jobs run once per retraining cycle and are validated end-to-end in the training notebook (PR-AUC 0.5786, lift 16.5×). Writing unit tests for them would duplicate the notebook without adding safety.

The serving path is tested aggressively because a regression there means wrong decisions on real traffic.

## Test files

### `tests/conftest.py` — fixtures (not a test file)

Two shared fixtures used across files:

- `sample_gold_df`: 5-row DataFrame mimicking the Gold schema for tool tests
- `mock_agent_state`: valid `AgentState` dict for agent/reporter tests
- `api_client`: `TestClient(app)` with the entire `lifespan` mocked — Gold load (2s), SHAP explainer (7s), pickles (3s), Qdrant connection. Tests run in ~100ms each instead of ~12s.

### `tests/test_config.py` — 2 tests

Covers `ConfigLoader.get()`:
- Top-level keys (dict), nested keys (scalar), types
- Missing key returns `None` or `default`

**Protects**: config structure changes. If someone renames `split.test_start_day` or removes a section, tests fail before deployment.

### `tests/test_decisions.py` — 5 tests

Covers the core business logic:

- `_map_decision` — three zones (approve / review / block)
- `router_node` — decision → route mapping
- Boundary cases (`calibrated == threshold` → higher zone, not lower)
- `build_explain_key` — Redis key format (`explain:v1:{model_version}:{tx_id}`)
- `Evidence.value` accepts scalars (bool, int, float), rejects arrays

**Protects**:
- **Threshold logic**: `>=` vs `>` bugs are invisible on normal cases — boundary tests catch them.
- **Cache invalidation**: the key includes `model_version`, so retraining a model invalidates stale reports automatically. If the format changes, old reports become unreachable.
- **Evidence contract**: the LLM was previously allowed to put `[]` in `value` — this crashed Pydantic. The test locks the fix.

### `tests/test_llm_client.py` — 3 tests

Covers `invoke_with_fallback`:

1. Groq succeeds → `provider="groq"`
2. Groq raises `429` → falls back to Gemini, `provider="gemini"`
3. Groq raises a **non-429** error → re-raises, does **not** fall back

**Protects**: the resilience mechanism. Test 2 covers the exact behavior that reduced P99 from 114s to ~15s under Locust load. Test 3 is critical: falling back on **real errors** would mask bugs — an `AssertionError` or `ValueError` should surface, not silently retry on a different provider.

### `tests/test_tools.py` — 3 tests

Covers deterministic tools + reporter fallback:

- `get_transaction_frame(unknown_id)` → `ValueError` (drives HTTP 404)
- `get_customer` on a new customer → `is_new_customer=True`, `n_transactions=0`
- `_fallback_report` → `confidence="low"`, `audit_id` prefixed `fallback-`, single evidence item

**Protects**: the 404 path on `/predict`, the most common business case in fraud (a new customer), and the reporter's graceful degradation when the LLM fails.

### `tests/test_api.py` — 6 tests

Endpoints via `TestClient`, lifespan fully mocked:

1. `GET /health` → 200 `{"status": "ok"}`
2. `GET /ready` → 200 with all checks OK
3. `GET /ready` → 503 when Qdrant unreachable
4. `POST /predict` → 200, correct JSON, **no `shap_factors`**
5. `POST /predict` on unknown ID → 404
6. `POST /explain` with cache HIT → 200, **LLM never called**

**Protects**:
- **Kubernetes probes**: `/health` and `/ready` are the difference between a pod receiving traffic and a pod being killed.
- **Graceful degradation**: when Qdrant is down, `/ready` returns 503 — Azure stops sending traffic to that pod instead of letting `/explain` fail.
- **Architectural decision**: SHAP was removed from `/predict` (it was the Locust bottleneck at 100 users). Test 4 prevents regression.
- **Cache effectiveness**: Test 6 uses `side_effect=AssertionError("LLM must not be called")` on `investigate_transaction`. If the cache breaks, the test fails loudly instead of silently spending tokens.

### `tests/test_integration.py` — 3 tests

- **Fast path skips LLM**: `build_graph().invoke({decision:"approve"})` → `fast_approve` route, `investigator_node` and `reporter_node` never called (mocked as `AssertionError`).
- **Real scoring pipeline**: `score_transaction(3433766)` with **no mocks** on the scoring chain — loads Gold, 3 pickles, runs TargetEncoder + LightGBM + Platt calibrator. Asserts invariants (`raw > calibrated`, `decision == _map_decision(calibrated, thresholds)`), not exact values.
- **Evidence prompt contract**: `format_evidence_for_prompt` emits the 6 sections (`TRANSACTION`, `MODEL OUTPUT`, `CUSTOMER`, `RELATED_TRANSACTIONS`, `SHAP_FACTORS`, `SIMILAR_CASES`).

**Protects**:
- **Cost control**: an `approve` decision must never reach the LLM. If the router breaks, every normal transaction pays Groq tokens.
- **Pipeline integrity**: the only test that loads the real Gold and pickles. Catches silent regressions — encoder reorders columns, calibrator not applied, thresholds stale.
- **Prompt drift**: if a section disappears, the LLM receives a truncated context and quality degrades silently. The test locks the contract.

## Coverage analysis
Total: 46% (1,371 statements, 738 missed)


The number is misleading. Breakdown:

**✅ Critical modules at >85%** (12 files):
- `router.py`, `schemas.py`, `state.py`, `graph.py`, `metrics.py` → **100%**
- `app/schemas.py`, `health.py`, `logger.py` → **100%**
- `main.py` (94%), `prompts.py` (91%), `predict.py` (88%), `api_logger.py` (88%)

**🟡 Partially covered (36–74%)** — missing branches require a real LLM call (~5,000 tokens per test run, unacceptable in CI):
- `fraud_investigator.py` (51%): `score_transaction` covered, `investigate_transaction` not
- `explain.py` (52%): cache HIT covered, cache MISS not
- `cache.py` (56%): happy path covered, Redis error branches not
- `llm_client.py` (70%): 3 fallback branches covered, semaphore not
- `reporter.py` (42%), `investigator.py` (36%): LLM nodes not covered directly

**❌ Offline jobs at 0%** (500+ missed statements) — validated in notebook, not in unit tests:
- `trainer.py`, `evaluator.py`, `splitter.py`, `ingest.py`, `indexer.py`, `shap_explainer.py`, `tools_similar.py`, `validation/schemas.py`

## What is NOT tested (and why)

- **The real LLM response quality** — subjective, evaluated separately via DeepEval (LLM-as-a-judge, reference-free). A unit test cannot assert "the report is faithful".
- **The training pipeline** — validated end-to-end in the notebook: PR-AUC 0.5786 holdout, lift 16.5× vs random.
- **The offline indexer** — one-shot job, already run in prod (17,889 points in Qdrant).
- **The graph assembly itself** — `build_graph()` is deterministic; the individual nodes are tested and the integration test exercises the full path for `approve`.
- **The `@tool` decorators themselves** — LangChain framework code, not ours.

## Regression guards — tests born from real bugs

| Bug found during dev | Test that now protects it |
|----------------------|---------------------------|
| `Evidence.value` rejected `True` (bool) | `test_evidence_item_accepts_scalar_rejects_array` |
| `_map_decision` used `>` instead of `>=` | `test_map_decision_includes_boundaries` |
| Cache key did not include `model_version` | `test_build_explain_key_format` |
| Fallback triggered on non-429 errors | `test_raises_on_non_rate_limit_error` |
| `shap_factors` crept back into `/predict` | `test_predict_ok` (asserts absence) |
| Config thresholds stale (`0.2753` vs `0.7346`) | `test_score_transaction_real_pipeline` (asserts decision coherence) |

## Running the tests

```bash
# Full suite
pytest tests/ -v

# Single file
pytest tests/test_decisions.py -v

# With coverage HTML report
pytest tests/ --cov=app --cov=src --cov-report=html --cov-fail-under=0
start htmlcov/index.html

# With test report
pip install pytest-html
pytest tests/ --html=report.html --self-contained-html
# Agent evaluation — 2-level assessment

**Date**: 2026-11-11
**Agent**: LangGraph multi-agent pipeline (retrieval → investigator → reporter)
**LLM**: Groq `gpt-oss-20b` (primary), Gemini `3.5-flash-lite` (fallback + judge)
**Dataset**: 10 transactions `review` from the test set (5 frauds, 5 legitimes)

## Why only 2 levels?

The agent does NOT decide. The decision is taken upstream by a deterministic
pipeline: model.predict_proba → calibrator → _map_decision(thresholds) → router_node → fast_path OR full_investigation


On `/explain`, `model_decision` is already `"review"` before the LLM runs.
The agent's role is to **explain** that decision with evidence, not to
change it. Evaluating "recommendation accuracy" would test a decision the
agent never makes.

The two meaningful metrics are therefore:
1. **Q1 — Retrieval correctness**: are the tools returning the right data?
2. **Q2 — Hallucination**: does the narrative contradict the evidence?

## Q1 — Retrieval correctness (deterministic)

For each of the 10 transactions, 4 checks are run against the Gold Parquet:

| Check | What it verifies |
|---|---|
| `check_transaction` | `amount`, `hour_of_day`, `uid` match the Gold row |
| `check_customer` | `n_transactions` equals direct Gold count; `is_new_customer` coherent |
| `check_shap` | 5 features in `feature_names_out`; rank 1 = max \|shap\|; direction consistent |
| `check_similar_cases` | 5 items; `transaction_dt < train_end_dt` (anti-leakage); `score >= 0.6` |

**Result**: 40/40 checks pass → **100% pass rate**.

This is a **non-regression test**: it fails the moment a tool returns wrong
data, without requiring any LLM call.

## Q2 — Hallucination detection (Gemini judge)

For each of the 10 narratives (2500 chars produced by the investigator),
a Gemini judge answers 5 binary questions:

| # | Question | What it detects |
|---|---|---|
| Q1 | Do numeric values cited match the evidence? | Invented amounts, velocities, SHAP values |
| Q2 | Do customer claims match the evidence? | Invented customer history |
| Q3 | Do similar_cases cited exist in the evidence? | Invented Qdrant matches |
| Q4 | Is the recommendation justified by the evidence? | Unsupported conclusions |
| Q5 | Is the narrative free of internal contradictions? | E.g. amount 554 in one section, 850 in another |

**Score** = number of `false` answers / 5
- `0.0` = fully faithful
- `1.0` = entirely fabricated

**Result**: mean score **0.06** over 10 cases, **0 cases > 0.5**.

This corresponds to ~3 unsupported claims across 10 × 5 = 50 judgments,
i.e. **94% faithfulness**.

## What is NOT evaluated (and why)

### Recommendation accuracy
The decision is deterministic upstream (model + thresholds + router).
The agent's role is explanatory. See "Why only 2 levels?" above.

### Narrative quality (style, coherence beyond contradictions)
Subjective. LLM-as-judge can measure contradictions, but "is this a good
explanation?" has no ground truth. Deferred to a future DeepEval pass.

### Investigation depth across the full test set
30-case evaluation would be statistically significant. 10 cases give a
trend, not a confidence interval. The pipeline is reproducible: bump
`n_fraud`/`n_legit` in `AgentEvaluator.run()` to scale up.

## What this evaluation demonstrates

- **Zero retrieval bugs**: 40 deterministic assertions pass
- **Zero hallucination hotspots**: 94% faithfulness across 10 long narratives
- **Reproducible**: `python -m src.agents.evaluation.evaluator` regenerates
  the results in ~5 min (with rate-limit handling)
- **Auditable**: every claim can be traced back to a tool output via the
  `Evidence` Pydantic contract

## Files

- `src/agents/evaluation/retrieval_checker.py` — Q1 implementation
- `src/agents/evaluation/hallucination_judge.py` — Q2 implementation
- `src/agents/evaluation/evaluator.py` — orchestrator + CLI
- `reports/evaluation/agent_<timestamp>.json` — raw results (gitignored)



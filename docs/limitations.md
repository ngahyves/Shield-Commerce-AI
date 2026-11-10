# Known limitations

## /predict — TargetEncoder latency

`sklearn.preprocessing.TargetEncoder` is optimized for batch transformation,
not single-row inference. On a single transaction, it costs ~27-55 ms
regardless of hardware. This is the dominant cost of `/predict`.

**Why not fix it**: replacing the encoder would invalidate the trained model
(`target_encoder.pkl` + `classifier.pkl`). Documented as accepted technical debt.

## /predict — Throughput ceiling

Limited by RAM: each uvicorn worker loads the 636 MB Gold DataFrame.
2 workers = 1.4 GB, which is the max on an 8 GB dev laptop.

**Measured baselines** (see `baselines/summary.csv`):

| Version | Workers | Encoder | P50 | Req/s | Failures |
|---------|---------|---------|-----|-------|----------|
| v1 | 1 | sklearn | 8 400 ms | 9.9 | 0.93% |
| **v2**  | **2**   | **sklearn** | **2 900 ms** | **25.0** | **0%** |
| v3  | 2 | ONNX (8 threads) | 3 300 ms | 22.5 | 0% |
| v3b | 2 | ONNX (1 thread)  | 3 700 ms | 21.2 | 0% |

**v2 is the reference baseline.**

## ONNX encoder — tested and reverted

### What was tested

Both `TargetEncoder` and `OrdinalEncoder` were converted to ONNX format
(`skl2onnx` + `onnxruntime`). ONNX Runtime executes the encoders in C++
and releases the GIL during `sess.run()`.

### Results

**Isolated benchmark** (1 row, warm):

| Encoder | Latency | Speedup |
|---------|---------|---------|
| sklearn | 27-55 ms | reference |
| ONNX (1 thread/session) | 5.6-19 ms | 2-4.9× |

**Under load** (Locust, 100 users, 2 workers):

| Version | P50 | Req/s | vs v2 |
|---------|-----|-------|-------|
| v2 (sklearn) | 2 900 ms | 25.0 | — |
| v3 (ONNX 8 threads) | 3 300 ms | 22.5 | -15% |
| v3b (ONNX 1 thread) | 3 700 ms | 21.2 | -28% |

### Root cause

The Python wrapper around ONNX sessions performs ~15 GIL-bound operations
per request (`astype(str)`, `isna().to_numpy()`, `np.hstack()`,
`pd.DataFrame()`, etc.). Under 8 concurrent threads per worker, these
serialize and add contention that does not exist with sklearn's single
Cython `transform()` call (which holds and releases the GIL once).

### Decision

Reverted to sklearn. The ONNX code is kept behind `ENABLE_ONNX=1`
(`src/agents/fraud_investigator.py`) for a future **single-graph ONNX
pipeline** (encoder + LightGBM combined), which would eliminate the
Python wrapper entirely.

### Lesson

Micro-benchmarks don't predict load performance when Python overhead is
on the critical path. Measure under production-like conditions before
shipping an optimization.

## /explain — LLM-bound

`/explain` runs the full agentic pipeline (retrieval + investigator LLM +
reporter LLM). It requires ~3600 tokens per call across 2 Groq API calls.

**Free tier quota**: Groq = 8,000 TPM → ~2 `/explain`/min maximum.
**Mitigation**: semaphore (max 2 concurrent) + fallback to Gemini (separate quota).
**Observed P99**: ~15s with fallback active (was 114s without).

**Production-grade** would require Groq Dev Tier ($5/mo, 100K+ TPM) or
self-hosted LLM. This is a quota limitation, not a code limitation.

## Production paths not implemented

Three levers could go beyond the v2 baseline (25 req/s):

1. **mmap Arrow** — share the Gold DataFrame across workers via memory-mapped
   Arrow table. Reduces RAM per worker from ~700 MB to ~100 MB, enabling
   8 workers on 8 GB instead of 2. Expected: ~96 req/s, P50 ~800 ms.

2. **Redis feature store** — pre-encode all 590K transactions once and serve
   from Redis. Only helps re-evaluations (audit, retry), not real-time
   since each new transaction is unique.

3. **Batch endpoint** — amortize the TargetEncoder cost over N rows.
   Single row: 50 ms. 256 rows: ~0.3 ms/row. Useful for offline scoring,
   not for real-time.

## /predict — real-time vs replay

The current API takes a `transaction_id` and looks it up in the Gold
Parquet — a **replay** design that simulates real-time on historical data.

In production, the API would receive the **raw payload** (amount, card_id,
product_type, ~60 fields) and construct the 126 features on the fly:
- ~60 raw fields from the payload
- ~15 client aggregates from a feature store (Redis) — `uid_amount_mean`,
  `velocity_count_24h`, etc.
- 7 target-encoded columns (requires the encoder, incompressible 50 ms)

Supporting raw-payload inference is out of scope for this project (dataset
is offline). Documented as a future production path.
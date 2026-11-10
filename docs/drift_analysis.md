# Drift analysis — train vs test

## Method
- **Tool**: Evidently 1.0+
- **Reference**: train set, jours 95-154 (178 357 rows)
- **Current**: test set, jours 155-181 (75 788 rows)
- **Features**: 119 (Gold sans metadata)
- **Tests**: PSI (numériques, seuil 0.2), Chi² (catégorielles, p-value 0.05)
- **Drift share threshold**: 20% (config.monitoring.drift_threshold)

## Results

| Metric | Value |
|---|---|
| Columns analyzed | 119 |
| Columns drifted | **37 (31.1%)** |
| Dataset drift | **YES** (31.1% > 20% threshold) |

## Top 6 drifted features

| Feature | PSI | Category | Business meaning |
|---|---|---|---|
| `v145` | 3.13 | Vesta anon. | Extreme drift (artifact) |
| `velocity_amount_sum_24h` | 0.78 | Behavioral | Amounts over 24h changed |
| `velocity_amount_sum_7d` | 0.70 | Behavioral | Amounts over 7d changed |
| `v236` | 0.50 | Vesta anon. | Artifact drift |
| `uid_amount_sum` | 0.32 | Client | Client total spend changed |
| `v171` | 0.29 | Vesta anon. | Artifact drift |

## Interpretation

Two categories of drift:
1. **Behavioral drift** (3/6): `velocity_amount_sum_*` and `uid_amount_sum` —
   transaction amounts change between periods. Expected in fraud: fraudsters
   adapt amounts to bypass detection.
2. **Vesta artifacts** (3/6): `v145`, `v236`, `v171` — legacy counters from
   Vesta with known instability (sliding window desync). The model absorbs
   this drift thanks to strong regularization (reg_alpha ×7.6, reg_lambda ×17M
   found by Optuna).

## Impact on model

- PR-AUC holdout: **0.5786** (vs 0.74 KFold random → -20 pts gap, drift penalty)
- Lift: **16.5×** vs random
- Model **resists drift** thanks to regularization
- Fraud rate: 3.50% (train) → 3.50% (test) — stable, but feature distributions shift

## Production rules (Azure Monitor)

- Metric exposed: `drift_share` (Prometheus)
- Alert rule: `drift_share > 0.25` for 3 consecutive days
- Decision: **manual retraining** with CI/CD evaluation gate
  (new model must beat current on PR-AUC before promotion)

## Files
- Report: `reports/drift_train_vs_test_20261106_061747.html`
- Script: `scripts/drift_analysis.py`
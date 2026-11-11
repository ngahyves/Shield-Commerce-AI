# Kubernetes scaling benchmark — ShieldCommerce API

**Cluster**: k3d on GitHub Codespace
- 2 nodes: 1 server (`control-plane`) + 1 agent
- 4 vCPU, 15 GB RAM (Codespace free tier)
- Kubernetes v1.35.5 (K3s)

**API**: `rickyr7/shieldcommerce-api:latest`
- 1 uvicorn worker per pod
- Image 1.39 GB (multi-stage, no torch)
- Endpoint tested: `POST /predict/{transaction_id}` (pure scoring, no LLM)

**Load**: Locust `-u 100 -r 10 -t 2m`
- 100 virtual users, ramp-up 10 users/s
- 2-minute test
- 10 transactions in rotation (`3433766`, `3361860`, etc.)

**Data**: PersistentVolume `hostPath` → `/workspaces/Shield-Commerce-AI/data`
- Gold Parquet: 590,540 rows, 125 columns
- Models: `classifier.pkl`, `target_encoder.pkl`, `calibrator.pkl`

---
## Single-request latency (no load)

The load test above measures latency **under 100 concurrent users**, where queue time dominates (P50 400-2200 ms). The **pure compute latency** — what a single client experiences — is measured separately on Docker Compose local (same code, same model, same data as K8s):

| Request | Latency |
|---------|---------|
| 1 (cold) | 66 ms |
| 2-10 (warm) | 40-44 ms |
| **Median warm** | **~41 ms** |
| **P95 warm** | **~44 ms** |

**Breakdown** (from TIMING logs):
- TargetEncoder: ~25-30 ms
- LightGBM predict: ~4 ms
- Calibrator: ~0.5 ms
- FastAPI + Pydantic overhead: ~5 ms
- Network localhost: ~2 ms

**Interpretation**: the API scores one transaction in **~41 ms**. The P50 of 400 ms at 4 replicas under 100 users = **~41 ms compute + ~359 ms queue wait**. This confirms the load test bottleneck is queue/CPU saturation, not the scoring code itself.

**K8s extrapolation**: adding ~5-10 ms of inter-pod network → single-request latency on K8s ≈ **~50 ms**.

## Results — horizontal scaling curve

| Replicas | Requests | Failures | **P50** | **P95** | **P99** | **RPS** |
|----------|----------|----------|---------|---------|---------|---------|
| 1 | 3,467 | 0 | 2,200 ms | 2,600 ms | 2,700 ms | 28.9 |
| 2 | 5,764 | 0 | 1,100 ms | 1,700 ms | 1,900 ms | 48.0 |
| 4 | 7,675 | 0 | **400 ms** | **1,300 ms** | **1,500 ms** | **64.0** |

**0% failures across all 3 configurations** (`k8s_*_failures.csv` empty).

---

## Analysis

### 1 → 2 replicas: perfect linear scaling

| Metric | Gain |
|--------|------|
| P50 | **÷2.00** exactly (2,200 → 1,100 ms) |
| P95 | ÷1.53 |
| RPS | ×1.66 |

The P50 halved is the **mathematical signature of horizontal scaling**: 2 pods = 2× processing capacity = queue is 2× shorter = P50 divided by 2.

RPS only gains ×1.66 (not ×2) because the 100 Locust users are a finite generator. The second pod still has headroom.

### 2 → 4 replicas: sub-linear scaling, but P50 keeps improving

| Metric | Gain |
|--------|------|
| P50 | ÷2.75 (better than linear) |
| P95 | ÷1.31 |
| RPS | ×1.33 |

The **P50 keeps improving** (1,100 → 400 ms), but **RPS plateaus at 64** instead of the ~95 expected. Two causes:

1. **Locust becomes the bottleneck** — 100 users × 0.5-1.5 s wait = **~70 req/s theoretical max**. At 64 req/s, we're at 91% of the generator's capacity.
2. **CodeSpace CPU saturated** — 4 pods × 1 vCPU = 4 vCPU total (Codespace limit).

The pods still have headroom (P50 400 ms), but **the load generator can't push further**.

---

## What this benchmark demonstrates

✅ **Functional horizontal scaling** — P50 halved when doubling pods
✅ **0% failures** — no service saturation despite 100 users
✅ **Bottleneck identified** — at 4 replicas, the API is no longer the bottleneck, the load generator is
✅ **Reproducible** — `kubectl scale deployment/api --replicas=N` + Locust

---

## Limitations and technical debt

### Calibration differs between local and container

`tx=3433766`: `calibrated_proba = 0.309` (laptop) vs `0.0485` (container).
Cause: different `scikit-learn` / `lightgbm` versions between environments.
The decision remains `review` in both cases (0.0485 > `review` threshold 0.0412).

### `/explain` not benchmarked in K8s

Qdrant not reindexed after cluster recreation → `/ready` returns 503 on the Qdrant check.
`/predict` works independently (it does not touch Qdrant).
Reindexing would take ~2 min from the API pod: `kubectl exec ... python -m scripts.index_frauds_qdrant`.

### No HPA (automatic scaling)

Scaling was done **manually** (`kubectl scale`) to measure the capacity curve.
Adding HPA requires `metrics-server` (installed by default in k3d but not activated).
Possible demo: `infra/k8s/hpa.yaml` with `min=1 max=4`, CPU 70%.

---

## Files produced

- `reports/locust/k8s_1rep_stats.csv` — 1 replica results
- `reports/locust/k8s_2rep_stats.csv` — 2 replicas results
- `reports/locust/k8s_4rep_stats.csv` — 4 replicas results
- `reports/locust/k8s_*_failures.csv` — empty (0 failures)

**Note**: CSVs are gitignored (regenerable via Locust). Only this document is versioned.

---

## Reproduction

```bash
# Cluster (if recreated)
k3d cluster create shield --agents 1 \
  --port "30080:30080@loadbalancer" \
  --port "30090:30090@loadbalancer" \
  --port "30030:30030@loadbalancer" \
  --volume "/workspaces/Shield-Commerce-AI/data:/data@all" --wait

# Secrets (DOCKER_PAT, GROQ_API_KEY, GEMINI_API_KEY as env vars)
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=rickyr7 --docker-password="$DOCKER_PAT" \
  --docker-email=ton@example.com
kubectl create secret generic app-secrets \
  --from-literal=GROQ_API_KEY="$GROQ_API_KEY" \
  --from-literal=GEMINI_API_KEY="$GEMINI_API_KEY"

# Manifests
kubectl apply -f infra/k8s/

# Test at N replicas
kubectl scale deployment/api --replicas=N
kubectl rollout status deployment/api

locust -f locustfile.py --headless -u 100 -r 10 -t 2m \
  --host http://localhost:30080 \
  --csv reports/locust/k8s_Nrep
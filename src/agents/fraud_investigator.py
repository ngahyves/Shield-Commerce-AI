"""Fraud investigation agent — production entry point.

Single public contract: `investigate_transaction(transaction_id) -> dict`.
The API calls this function. Batch jobs that have already scored transactions
should call `build_graph().invoke({...})` directly instead of paying the
re-scoring cost.

The CLI at the bottom is a development tool only.
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal
import numpy as np

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.agents.graph import build_graph
from src.agents.tools import get_transaction_frame, _shap_explainer
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

load_dotenv()
log = ProjectLogger(__name__).get_logger()


@lru_cache(maxsize=1)
def _load_scoring_artifacts():
    """Load classifier, encoder, calibrator, thresholds once. Cached."""
    from src.calibration.calibrator import ProbabilityCalibrator
    from src.features.target_encoder import TargetEncoderBuilder

    cfg = ConfigLoader()
    models_dir = Path(cfg.get("paths", "models"))

    log.info("loading scoring artifacts (first call only)")

    with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # ONNX disabled — see docs/limitations.md
    # ONNX was 2x faster isolated but 28% slower under load (GIL contention).
    # Code kept below for reference. Set ENABLE_ONNX=1 in .env to re-enable.
    onnx_sessions = None
    # import os
    # if os.getenv("ENABLE_ONNX", "0") == "1":
    #     ord_path = models_dir / "ordinal_encoder.onnx"
    #     te_path = models_dir / "target_encoder.onnx"
    #     if ord_path.exists() and te_path.exists():
    #         from onnxruntime import InferenceSession, SessionOptions
    #         opts = SessionOptions()
    #         opts.intra_op_num_threads = 1
    #         opts.inter_op_num_threads = 1
    #         onnx_sessions = {
    #             "ord": InferenceSession(str(ord_path), sess_options=opts),
    #             "te": InferenceSession(str(te_path), sess_options=opts),
    #         }
    #         log.info("onnx sessions loaded (1 thread each)")

    return {
        "model": joblib.load(models_dir / "classifier.pkl"),
        "encoder": TargetEncoderBuilder.load(models_dir / "target_encoder.pkl"),
        "onnx_sessions": onnx_sessions,
        "calibrator": ProbabilityCalibrator.load(models_dir / "calibrator.pkl"),
        "metadata_cols": cfg.get("features", "metadata_cols"),
        "thresholds": metadata["thresholds"],
        "model_version": metadata["model_version"],
        "model_name": cfg.get("model", "algorithm", default="lightgbm"),
    }
@lru_cache(maxsize=1)
def _get_graph():
    """Compile the investigation graph once. Cached."""
    return build_graph()

def _map_decision(calibrated: float, thresholds: dict) -> Literal["approve", "review", "block"]:
    """Map calibrated probability to a decision using computed thresholds."""
    if calibrated >= thresholds["block"]:
        return "block"
    if calibrated >= thresholds["review"]:
        return "review"
    return "approve"

def _encode_onnx(X: pd.DataFrame, arts: dict) -> pd.DataFrame:
    """Transform using ONNX sessions. Same output as TargetEncoderBuilder.transform().

    Order: [ordinal(cat_cols), passthrough(non-cat numeric), target(te_cols)].
    NaN in cat_cols are propagated (sklearn behaviour, patched in ONNX output).
    """
    import numpy as np

    encoder = arts["encoder"]          # for cat_cols_ and te_cols metadata
    sess = arts["onnx_sessions"]
    cat_cols = encoder.cat_cols_
    te_cols = encoder.te_cols

    # 1. Ordinal with NaN fix
    X_ord = sess["ord"].run(
        None, {"ord_inputs": X[cat_cols].astype(str).values}
    )[0]
    nan_mask = X[cat_cols].isna().to_numpy()
    X_ord[nan_mask] = np.nan

    # 2. Target
    X_te = sess["te"].run(
        None, {"te_inputs": X[te_cols].astype(str).values}
    )[0]

    # 3. Passthrough = everything not in cat_cols
    passthrough_cols = [c for c in X.columns if c not in cat_cols]
    X_num = X[passthrough_cols].values.astype(np.float32)

    # 4. Concat
    X_final = np.hstack([X_ord, X_num, X_te])

    cols = list(cat_cols) + passthrough_cols + [f"{c}_te" for c in te_cols]
    return pd.DataFrame(X_final, columns=cols, index=X.index)


def score_transaction(transaction_id: int) -> tuple[float, float, str]:
    df_row = get_transaction_frame(transaction_id)
    arts = _load_scoring_artifacts()

    X_input = df_row.drop(columns=arts["metadata_cols"], errors="ignore")

    if arts["onnx_sessions"] is not None:
        X = _encode_onnx(X_input, arts)
    else:
        X = arts["encoder"].transform(X_input)   

    raw = float(arts["model"].predict_proba(X)[0, 1])
    cal = float(arts["calibrator"].transform(np.array([raw]))[0])
    decision = _map_decision(cal, arts["thresholds"])
    return raw, cal, decision


def investigate_transaction(transaction_id: int) -> dict:
    """Self-contained investigation from a transaction id."""
    t0 = time.perf_counter()

    raw_score, calibrated, decision = score_transaction(transaction_id)
    arts = _load_scoring_artifacts()

    log.info(
        "scored: tx=%d raw=%.4f calibrated=%.4f decision=%s",
        transaction_id, raw_score, calibrated, decision,
    )

    final_state = build_graph().invoke({
        "transaction_id": transaction_id,
        "calibrated_proba": calibrated,
        "raw_score": raw_score,
        "model_decision": decision,
        "model_version": arts["model_version"],
    })

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "investigation complete: tx=%d route=%s total=%.0f ms",
        transaction_id, final_state.get("route"), total_ms,
    )
    return final_state


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.agents.fraud_investigator <transaction_id>")
        sys.exit(1)

    result = investigate_transaction(int(sys.argv[1]))
    report = result["report"]

    print(f"\nROUTE    : {result.get('route')}")
    print(f"DECISION : {report['model_decision']}")
    print(f"REPORT   : {report['recommendation']} (confidence: {report['confidence']})")
    print(f"REASONING: {report['reasoning'][:300]}...")

    shap = result.get("shap_factors") or []
    print(f"\nSHAP FACTORS ({len(shap)}):")
    for f in shap:
        feature = f.get("feature", "?")
        value = f.get("shap_value", 0.0)
        direction = f.get("direction", "")
        sign = "▲" if direction == "positive" else "▼"
        print(f"  {sign} {feature:<30} {value:+.4f}  ({direction})")

    print(f"\nEVIDENCE : {len(report['evidence'])} items")
    print(f"SIMILAR  : {len(report['similar_cases'])} cases")
    print(f"LATENCY  : {report['latency_ms']}")
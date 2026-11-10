"""Probability calibration for the fraud classifier.

Compares raw vs isotonic vs platt on the Brier score.
Fits on days 155-164, evaluates on days 165-181.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.features.target_encoder import TargetEncoderBuilder
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()

SECONDS_PER_DAY = 86_400


class ProbabilityCalibrator:
    def __init__(self, method: str = "raw"):
        assert method in ("raw", "isotonic", "platt"), f"unknown method: {method}"
        self.method = method
        self.calibrator_ = None

    def fit(self, y_val: np.ndarray, proba_val: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "raw":
            return self
        x = proba_val.reshape(-1, 1)
        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(proba_val, y_val)
        else:
            self.calibrator_ = LogisticRegression(C=1.0, max_iter=1000)
            self.calibrator_.fit(x, y_val)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.method == "raw":
            return proba
        if self.method == "isotonic":
            return self.calibrator_.predict(proba)
        return self.calibrator_.predict_proba(proba.reshape(-1, 1))[:, 1]

    def brier(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        return float(brier_score_loss(y_true, np.clip(proba, 0, 1)))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "calibrator": self.calibrator_}, path)
        return path
    
    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        payload = joblib.load(path)
        instance = cls(payload["method"])
        instance.calibrator_ = payload["calibrator"]
        return instance


if __name__ == "__main__":
    from src.ingestion.ingest import GoldIngestor
    from src.training.splitter import TemporalSplitter

    # 1. Config and paths
    cfg = ConfigLoader()
    models_dir = Path(cfg.get("paths", "models"))
    metadata_cols = cfg.get("features", "metadata_cols")
    target_col = cfg.get("features", "target_col")
    split_day = cfg.get("calibration", "split_day", default=165)

    # 2. Load model + encoder
    model = joblib.load(models_dir / "classifier.pkl")
    encoder = TargetEncoderBuilder.load(models_dir / "target_encoder.pkl")

    # 3. Load + split test set
    df = GoldIngestor().ingest()
    _, test_df = TemporalSplitter().split(df)
    cutoff = split_day * SECONDS_PER_DAY
    calib_df = test_df[test_df["transaction_dt"] < cutoff].copy()
    eval_df = test_df[test_df["transaction_dt"] >= cutoff].copy()

    log.info(
        "calibration: %d rows (%.3f%% fraud), eval: %d rows (%.3f%% fraud)",
        len(calib_df), 100 * calib_df[target_col].mean(),
        len(eval_df), 100 * eval_df[target_col].mean(),
    )

    # 4. Predict on both segments
    X_calib = encoder.transform(calib_df.drop(columns=metadata_cols))
    X_eval = encoder.transform(eval_df.drop(columns=metadata_cols))
    proba_calib = model.predict_proba(X_calib)[:, 1]
    proba_eval = model.predict_proba(X_eval)[:, 1]
    y_calib = calib_df[target_col].values
    y_eval = eval_df[target_col].values

    # 5. Compare 3 calibrators
    results = {}
    for method in ("raw", "isotonic", "platt"):
        cal = ProbabilityCalibrator(method).fit(y_calib, proba_calib)
        proba_eval_cal = cal.transform(proba_eval)
        results[method] = {
            "brier": cal.brier(y_eval, proba_eval_cal),
            "calibrator": cal,
        }

    # 6. Report
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON — Brier score")
    print("=" * 60)
    print(f"{'Method':<12} {'Brier':>10}")
    print("-" * 60)
    for method, r in results.items():
        print(f"{method:<12} {r['brier']:>10.5f}")

    config_method = cfg.get("calibration", "method", default="auto")
    if config_method in ("raw", "isotonic", "platt"):
        best = config_method
        log.info("calibrator forced by config: %s", best)
    else:
        best = min(results, key=lambda m: results[m]["brier"])
        log.info("calibrator auto-selected (best Brier): %s", best)
    print(f"\nBest method: {best} (Brier = {results[best]['brier']:.5f})")

    # 7. Save best calibrator
    cal_path = models_dir / "calibrator.pkl"
    results[best]["calibrator"].save(cal_path)
    log.info("best calibrator (%s) saved: %s", best, cal_path)

    # 8. Update metadata
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["calibration"] = {
        "method": best,
        "brier_before": results["raw"]["brier"],
        "brier_after": results[best]["brier"],
        "brier_improvement": results["raw"]["brier"] - results[best]["brier"],
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("metadata updated: %s", meta_path)
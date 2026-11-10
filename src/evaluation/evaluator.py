"""Fraud model evaluation: 18 metrics, two-cost thresholds, calibration-aware."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()

SECONDS_PER_DAY = 86_400


class FraudEvaluator:
    def __init__(self):
        cfg = ConfigLoader()
        self.models_dir = Path(cfg.get("paths", "models"))
        self.metadata_cols = cfg.get("features", "metadata_cols")
        self.target_col = cfg.get("features", "target_col")
        self.cost_fn_review = cfg.get("cost", "review", "fn_multiplier", default=10)
        self.cost_fp_review = cfg.get("cost", "review", "fp_multiplier", default=1)
        self.cost_fn_block = cfg.get("cost", "block", "fn_multiplier", default=3)
        self.cost_fp_block = cfg.get("cost", "block", "fp_multiplier", default=15)
        self.split_day = cfg.get("calibration", "split_day", default=165)

    def _at(self, y_true, y_proba, threshold, cost_fn, cost_fp):
        pred = (y_proba >= threshold).astype(int)
        _, fp, fn, _ = confusion_matrix(y_true, pred).ravel()
        return {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
            "f3": float(fbeta_score(y_true, pred, beta=3, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "cost": int(fn * cost_fn + fp * cost_fp),
            "n_predicted": int(pred.sum()),
        }

    def evaluate(self, y_true, y_proba):
        pr_auc = float(average_precision_score(y_true, y_proba))
        p, r, thr = precision_recall_curve(y_true, y_proba)
        n_pos = y_true.sum()
        tp = r[:-1] * n_pos
        fp = tp / np.maximum(p[:-1], 1e-10) - tp
        fn = n_pos - tp

        # Review: FN×10 + FP×1 → low threshold
        cost_review = fn * self.cost_fn_review + fp * self.cost_fp_review
        thr_review = float(thr[int(np.argmin(cost_review))])

        # Block: FN×3 + FP×15 → high threshold
        cost_block = fn * self.cost_fn_block + fp * self.cost_fp_block
        thr_block = float(thr[int(np.argmin(cost_block))])

        if thr_block <= thr_review:
            log.warning("block <= review, forcing 20%% gap")
            thr_block = min(thr_review * 1.20, 0.99)

        m_r = self._at(y_true, y_proba, thr_review, self.cost_fn_review, self.cost_fp_review)
        m_b = self._at(y_true, y_proba, thr_block, self.cost_fn_block, self.cost_fp_block)
        m_5 = self._at(y_true, y_proba, 0.5, self.cost_fn_review, self.cost_fp_review)

        return {
            "pr_auc": pr_auc,
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "lift": pr_auc / float(y_true.mean()),
            # Review
            "thr_review": thr_review,
            "precision_at_review": m_r["precision"],
            "recall_at_review": m_r["recall"],
            "f1_at_review": m_r["f1"],
            "f2_at_review": m_r["f2"],
            "f3_at_review": m_r["f3"],
            "mcc_at_review": m_r["mcc"],
            "cost_at_review": m_r["cost"],
            "n_predicted_review": m_r["n_predicted"],
            # Block
            "thr_block": thr_block,
            "precision_at_block": m_b["precision"],
            "recall_at_block": m_b["recall"],
            "f1_at_block": m_b["f1"],
            "f2_at_block": m_b["f2"],
            "f3_at_block": m_b["f3"],
            "mcc_at_block": m_b["mcc"],
            "cost_at_block": m_b["cost"],
            "n_predicted_block": m_b["n_predicted"],
            # 0.5 reference
            "thr_05": 0.5,
            "precision_at_05": m_5["precision"],
            "recall_at_05": m_5["recall"],
            "f1_at_05": m_5["f1"],
            "f2_at_05": m_5["f2"],
            "f3_at_05": m_5["f3"],
            "mcc_at_05": m_5["mcc"],
            "cost_at_05": m_5["cost"],
            "n_predicted_05": m_5["n_predicted"],
        }
    def print_report(self, m):
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)
        print(f"PR-AUC : {m['pr_auc']:.4f}")
        print(f"AUC-ROC: {m['roc_auc']:.4f}")
        print(f"Lift: {m['lift']:.2f}x")

        print("\n" + "-" * 80)
        print(f"{'Operating point':<22} {'thr':>7} {'P':>6} {'R':>6} {'F1':>6} {'F2':>6} {'F3':>6} {'MCC':>6} {'Cost':>9} {'#Pred':>8}")
        print("-" * 80)

        rows = [
            (f"Review (FN×{self.cost_fn_review}+FP×{self.cost_fp_review})", "review"),
            (f"Block  (FN×{self.cost_fn_block}+FP×{self.cost_fp_block})", "block"),
            ("0.5 (ref)", "05"),
        ]
        for label, key in rows:
            print(
                f"{label:<22} "
                f"{m[f'thr_{key}']:>7.4f} "
                f"{m[f'precision_at_{key}']:>6.3f} "
                f"{m[f'recall_at_{key}']:>6.3f} "
                f"{m[f'f1_at_{key}']:>6.3f} "
                f"{m[f'f2_at_{key}']:>6.3f} "
                f"{m[f'f3_at_{key}']:>6.3f} "
                f"{m[f'mcc_at_{key}']:>6.3f} "
                f"{m[f'cost_at_{key}']:>9,} "
                f"{m[f'n_predicted_{key}']:>8,}"
            )
        print("=" * 80)

    def update_metadata(self, m):
        meta_path = self.models_dir / "model_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        metadata["thresholds"] = {
            "block": m["thr_block"],
            "review": m["thr_review"],
        }
        metadata["cost_assumption"] = {
            "review": {"fn": self.cost_fn_review, "fp": self.cost_fp_review},
            "block": {"fn": self.cost_fn_block, "fp": self.cost_fp_block},
        }
        metadata["metrics"] = {
            "threshold_independent": {
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "lift": m["lift"],
            },
            "at_review": {
                "threshold": m["thr_review"],
                "precision": m["precision_at_review"],
                "recall": m["recall_at_review"],
                "f1": m["f1_at_review"],
                "f2": m["f2_at_review"],
                "f3": m["f3_at_review"],
                "mcc": m["mcc_at_review"],
                "cost": m["cost_at_review"],
                "n_predicted": m["n_predicted_review"],
            },
            "at_block": {
                "threshold": m["thr_block"],
                "precision": m["precision_at_block"],
                "recall": m["recall_at_block"],
                "f1": m["f1_at_block"],
                "f2": m["f2_at_block"],
                "f3": m["f3_at_block"],
                "mcc": m["mcc_at_block"],
                "cost": m["cost_at_block"],
                "n_predicted": m["n_predicted_block"],
            },
            "at_0_5": {
                "threshold": 0.5,
                "precision": m["precision_at_05"],
                "recall": m["recall_at_05"],
                "f1": m["f1_at_05"],
                "f2": m["f2_at_05"],
                "f3": m["f3_at_05"],
                "mcc": m["mcc_at_05"],
                "cost": m["cost_at_05"],
                "n_predicted": m["n_predicted_05"],
            },
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info("metadata updated: %s", meta_path)


if __name__ == "__main__":
    from src.calibration.calibrator import ProbabilityCalibrator
    from src.features.target_encoder import TargetEncoderBuilder
    from src.ingestion.ingest import GoldIngestor
    from src.training.splitter import TemporalSplitter

    cfg = ConfigLoader()
    models_dir = Path(cfg.get("paths", "models"))
    metadata_cols = cfg.get("features", "metadata_cols")
    target_col = cfg.get("features", "target_col")
    split_day = cfg.get("calibration", "split_day", default=165)

    model = joblib.load(models_dir / "classifier.pkl")
    encoder = TargetEncoderBuilder.load(models_dir / "target_encoder.pkl")
    calibrator = ProbabilityCalibrator.load(models_dir / "calibrator.pkl")
    log.info("calibrator loaded: method=%s", calibrator.method)

    df = GoldIngestor().ingest()
    _, test_df = TemporalSplitter().split(df)
    eval_df = test_df[test_df["transaction_dt"] >= split_day * SECONDS_PER_DAY].copy()
    log.info(
        "eval set: %d rows (%.3f%% fraud)",
        len(eval_df), 100 * eval_df[target_col].mean(),
    )

    X_eval = encoder.transform(eval_df.drop(columns=metadata_cols))
    y_eval = eval_df[target_col].values
    proba_raw = model.predict_proba(X_eval)[:, 1]
    proba_cal = calibrator.transform(proba_raw)

    evaluator = FraudEvaluator()
    m_raw = evaluator.evaluate(y_eval, proba_raw)
    m_cal = evaluator.evaluate(y_eval, proba_cal)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC — raw vs calibrated")
    print("=" * 60)
    print(f"{'Metric':<15} {'raw':>10} {'calibrated':>12} {'delta':>10}")
    print("-" * 60)
    for k in ("pr_auc", "roc_auc", "lift"):
        print(f"{k:<15} {m_raw[k]:>10.4f} {m_cal[k]:>12.4f} {m_cal[k] - m_raw[k]:>+10.4f}")

    pr_drop = m_raw["pr_auc"] - m_cal["pr_auc"]
    if pr_drop > 0.005:
        log.warning("calibration drops PR-AUC by %.4f — using raw", pr_drop)
        chosen_metrics = m_raw
        chosen_label = "raw"
    else:
        chosen_metrics = m_cal
        chosen_label = f"calibrated ({calibrator.method})"

    print(f"\nChosen: {chosen_label}")
    evaluator.print_report(chosen_metrics)
    evaluator.update_metadata(chosen_metrics)
    log.info("done")
"""SHAP explainability for the fraud classifier.

Computes global feature importance (mean |SHAP| over a sample) and local
explanations for individual transactions. Both on encoded features (what the
model actually sees).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


class ShapExplainer:
    def __init__(self):
        cfg = ConfigLoader()
        self.models_dir = Path(cfg.get("paths", "models"))
        self.metadata_cols = cfg.get("features", "metadata_cols")
        self.target_col = cfg.get("features", "target_col")

        self.model = joblib.load(self.models_dir / "classifier.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values for the positive class, shape (n, features)."""
        values = self.explainer.shap_values(X)
        if isinstance(values, list):
            return values[1]
        if values.ndim == 3:
            return values[:, :, 1]
        return values

    def global_importance(self, X_sample: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Mean |SHAP| per feature across the sample. Returns a ranked DataFrame."""
        values = self._shap_values(X_sample)
        mean_abs = np.abs(values).mean(axis=0)
        df = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df[["rank", "feature", "mean_abs_shap"]]

    def explain_transaction(self, X_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Top N SHAP contributors for a single transaction (X_row = 1-row DataFrame)."""
        values = self._shap_values(X_row)[0]
        feature_values = X_row.iloc[0].values
        df = pd.DataFrame({
            "feature": X_row.columns,
            "shap_value": values,
            "feature_value": feature_values,
        })
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).head(top_n)
        return [
            {
                "feature": str(row["feature"]),
                "shap_value": float(row["shap_value"]),
                "feature_value": float(row["feature_value"]) if pd.notna(row["feature_value"]) else None,
            }
            for _, row in df.iterrows()
        ]

    def save_global_plot(self, X_sample: pd.DataFrame, path: Path, top_n: int = 20) -> Path:
        """Save a bar plot of global feature importance."""
        import matplotlib.pyplot as plt

        importance = self.global_importance(X_sample, top_n=top_n)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1], color="#4C72B0")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title(f"Top {top_n} features — global importance")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("plot saved: %s", path)
        return path

    def update_metadata(self, importance: pd.DataFrame) -> None:
        """Write top 20 global feature importance to model_metadata.json."""
        meta_path = self.models_dir / "model_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        metadata["top_features"] = [
            {"rank": int(r["rank"]), "feature": r["feature"], "mean_abs_shap": round(float(r["mean_abs_shap"]), 6)}
            for _, r in importance.iterrows()
        ]
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
    sample_size = 1000

    # 1. Load data (eval segment only)
    df = GoldIngestor().ingest()
    _, test_df = TemporalSplitter().split(df)
    eval_df = test_df[test_df["transaction_dt"] >= split_day * 86_400].copy()

    encoder = TargetEncoderBuilder.load(models_dir / "target_encoder.pkl")
    X_eval = encoder.transform(eval_df.drop(columns=metadata_cols))
    y_eval = eval_df[target_col].values

    # 2. Sample for global importance
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_eval), size=min(sample_size, len(X_eval)), replace=False)
    X_sample = X_eval.iloc[idx]

    explainer = ShapExplainer()

    # 3. Global importance
    importance = explainer.global_importance(X_sample, top_n=20)
    print("\n" + "=" * 60)
    print(f"GLOBAL FEATURE IMPORTANCE — top 20 (sample = {len(X_sample)})")
    print("=" * 60)
    print(importance.to_string(index=False))

    # 4. Save plot + metadata
    explainer.save_global_plot(X_sample, models_dir / "shap_global_importance.png", top_n=20)
    explainer.update_metadata(importance)

    # 5. Local explanations on one fraud and one legit
    fraud_idx = np.where(y_eval == 1)[0][0]
    legit_idx = np.where(y_eval == 0)[0][0]

    print("\n" + "=" * 60)
    print("LOCAL — fraud transaction")
    print("=" * 60)
    for r in explainer.explain_transaction(X_eval.iloc[[fraud_idx]], top_n=5):
        print(f"  {r['feature']:<25} {r['shap_value']:+.4f}  (value={r['feature_value']})")

    print("\n" + "=" * 60)
    print("LOCAL — legit transaction")
    print("=" * 60)
    for r in explainer.explain_transaction(X_eval.iloc[[legit_idx]], top_n=5):
        print(f"  {r['feature']:<25} {r['shap_value']:+.4f}  (value={r['feature_value']})")
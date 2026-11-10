"""LightGBM training with MLflow tracking and model registry. Pure training only."""

import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier

from src.features.target_encoder import TargetEncoderBuilder
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

logger = ProjectLogger(__name__).get_logger()


class FraudTrainer:
    """Trains a LightGBM fraud classifier using pre-encoded features."""

    def __init__(self, run_name: str | None = None) -> None:
        self.config = ConfigLoader()
        self.params: dict = self.config.get("model", "params")
        self.version: str = self.config.get("model", "version")
        self.models_dir: Path = Path(self.config.get("paths", "models"))
        self.metadata_cols: list[str] = self.config.get("features", "metadata_cols")
        self.target_col: str = self.config.get("features", "target_col")
        self.experiment: str = self.config.get("mlflow", "experiment_name", default="fraud_detection")
        self.registry_name: str = self.config.get("mlflow", "registered_model_name", default="fraud_detector")
        self.tracking_uri: str = self.config.get("mlflow", "tracking_uri", default="http://localhost:5000")
        self.run_name: str = run_name or f"lgbm-v{self.version}"
        self.model_: LGBMClassifier | None = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        preprocessor_path: str | Path | None = None,
    ) -> LGBMClassifier:
        encoder = TargetEncoderBuilder.load(
            preprocessor_path or self.models_dir / "target_encoder.pkl"
        )

        X_train = train_df.drop(columns=self.metadata_cols)
        y_train = train_df[self.target_col]
        X_val = val_df.drop(columns=self.metadata_cols)
        y_val = val_df[self.target_col]

        logger.info("transforming features...")
        X_train_enc = encoder.transform(X_train)
        X_val_enc = encoder.transform(X_val)

        logger.info("fitting lightgbm on %d features...", X_train_enc.shape[1])
        self.model_ = LGBMClassifier(**self.params)
        self.model_.fit(X_train_enc, y_train)

        # Ranking metrics — PR-AUC, ROC-AUC
        y_proba_val = self.model_.predict_proba(X_val_enc)[:, 1]
        pr_auc = float(average_precision_score(y_val, y_proba_val))
        roc_auc = float(roc_auc_score(y_val, y_proba_val))
        lift = pr_auc / float(y_val.mean())

        logger.info("val pr_auc=%.4f roc_auc=%.4f lift=%.2fx", pr_auc, roc_auc, lift)

        run_id, model_uri = self._log_mlflow(
            X_train_enc, X_val_enc, {"pr_auc": pr_auc, "roc_auc": roc_auc, "lift": lift}
        )
        self._save(X_train_enc.columns.tolist(), run_id, model_uri)
        return self.model_

    def _log_mlflow(
        self,
        X_train_enc: pd.DataFrame,
        X_val_enc: pd.DataFrame,
        metrics: dict,
    ) -> tuple[str, str]:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

        with mlflow.start_run(run_name=self.run_name) as run:
            mlflow.log_params({f"model_{k}": v for k, v in self.params.items()})
            mlflow.log_params({
                "train_rows": len(X_train_enc),
                "val_rows": len(X_val_enc),
            })
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(
                self.model_,
                artifact_path="classifier",
                registered_model_name=self.registry_name,
                input_example=X_val_enc.head(3),
            )
            return run.info.run_id, f"runs:/{run.info.run_id}/classifier"

    def _save(self, feature_names: list[str], run_id: str, model_uri: str) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, self.models_dir / "classifier.pkl")

        metadata = {
            "model_version": self.version,
            "mlflow_run_id": run_id,
            "mlflow_model_uri": model_uri,
            "feature_names_out": feature_names,
        }
        with open(self.models_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("model saved locally to %s", self.models_dir / "classifier.pkl")


if __name__ == "__main__":
    from src.ingestion.ingest import GoldIngestor
    from src.training.splitter import TemporalSplitter

    df = GoldIngestor().ingest()
    train_df, test_df = TemporalSplitter().split(df)

    t0 = time.time()
    trainer = FraudTrainer(run_name="lgbm-optuna-v1")
    trainer.train(train_df, test_df)
    elapsed = time.time() - t0

    print(f"\ntraining done in {elapsed:.0f}s")
    print(f"model: {trainer.models_dir / 'classifier.pkl'}")
    print(f"mlflow run: {trainer.run_name}")
"""Target encoding preprocessor. Fits on train, transforms any set."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


class TargetEncoderBuilder:
    """ColumnTransformer with OrdinalEncoder (all cats) + TargetEncoder (7 cols).

    Output = ordinal(cat_cols) + TE(te_cols) as separate _te columns + numeric passthrough.
    Reproduces the notebook architecture: both encodings coexist.
    """

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self.te_cols: list[str] = cfg.get("features", "te_cols")
        self.metadata_cols: list[str] = cfg.get("features", "metadata_cols")
        self.te_params: dict = cfg.get("target_encoding")
        self.ordinal_params: dict = cfg.get("ordinal_encoding")
        self.models_dir: str = cfg.get("paths", "models")

        self.transformer_: ColumnTransformer | None = None
        self.te_transformer_: TargetEncoder | None = None
        self.cat_cols_: list[str] | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TargetEncoderBuilder":
        self.cat_cols_ = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

        log.info(
            "fitting encoder: %d ordinal cols + %d TE cols + %d numeric passthrough",
            len(self.cat_cols_),
            len(self.te_cols),
            len(X_train.columns) - len(self.cat_cols_),
        )

        # Ordinal encoding for ALL categorical columns (including TE cols)
        self.transformer_ = ColumnTransformer(
            transformers=[
                ("ord", OrdinalEncoder(dtype=np.float32, **self.ordinal_params), self.cat_cols_),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.transformer_.set_output(transform="pandas")
        self.transformer_.fit(X_train, y_train)

        # Target encoding for the 7 TE cols (produces 7 extra _te columns)
        self.te_transformer_ = TargetEncoder(**self.te_params)
        self.te_transformer_.fit(X_train[self.te_cols], y_train)

        log.info(
            "fit complete — output: %d columns",
            len(self.transformer_.get_feature_names_out()) + len(self.te_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transformer_ is None or self.te_transformer_ is None:
            raise RuntimeError("Encoder not fitted. Call fit() or load() first.")

        # 1. Ordinal + passthrough (all columns)
        X_encoded = self.transformer_.transform(X)

        # 2. TE columns (7 extra)
        te_values = self.te_transformer_.transform(X[self.te_cols])
        te_df = pd.DataFrame(
            te_values,
            columns=[f"{c}_te" for c in self.te_cols],
            index=X_encoded.index,
        )

        # 3. Concatenate: ordinal + numeric + TE
        return pd.concat([X_encoded, te_df], axis=1)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or f"{self.models_dir}/target_encoder.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "transformer": self.transformer_,
                "te_transformer": self.te_transformer_,
                "cat_cols": self.cat_cols_,
                "te_cols": self.te_cols,
            },
            path,
        )
        log.info("encoder saved: %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TargetEncoderBuilder":
        instance = cls()
        payload = joblib.load(path)
        instance.transformer_ = payload["transformer"]
        instance.te_transformer_ = payload["te_transformer"]
        instance.cat_cols_ = payload["cat_cols"]
        instance.te_cols = payload["te_cols"]
        log.info("encoder loaded: %s", path)
        return instance
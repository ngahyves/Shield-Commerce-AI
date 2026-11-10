"""Gold data ingestion: load, cast, validate.

Single responsibility: bring the Gold Parquet into memory with correct dtypes
and validated schema. NO feature engineering, NO temporal split.
"""

from pathlib import Path
import pandas as pd

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger
from src.validation.schemas import FraudSchemaValidator

logger = ProjectLogger(__name__).get_logger()


class GoldIngestor:
    """Load the Gold Parquet file, cast dtypes, validate with Pandera."""

    def __init__(self, parquet_path: str | None = None) -> None:
        """Initialize the ingestor.

        Args:
            parquet_path: Optional override. Defaults to config paths.gold_parquet.
        """
        self.config = ConfigLoader()
        self.parquet_path = Path(
            parquet_path or self.config.get("paths", "gold_parquet")
        )
        self.validator = FraudSchemaValidator()

    def ingest(self) -> pd.DataFrame:
        """Full pipeline: load → cast → validate.

        Returns:
            Validated DataFrame with downcast dtypes.
        """
        df = self._load()
        self._validate(df)          # validate on raw dtypes
        df = self._cast_dtypes(df)  # then optimize memory
        return df

    def _load(self) -> pd.DataFrame:
        """Load the Parquet file."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Gold Parquet not found: {self.parquet_path.resolve()}. "
                f"Export it from Databricks first (see docs/export_gold.md)."
            )
        logger.info("Loading Gold Parquet: %s", self.parquet_path)
        df = pd.read_parquet(self.parquet_path)
        logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
        return df

    def _cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast dtypes for memory efficiency."""
        logger.info("Casting dtypes for memory efficiency...")
        keep_int64 = {"transaction_id", "transaction_dt", "is_fraud"}

        for c in df.columns:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
            elif df[c].dtype == "int64" and c not in keep_int64:
                df[c] = df[c].astype("int32")
            elif df[c].dtype == "object":
                df[c] = df[c].astype("category")

        logger.info("Dtypes cast complete")
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate with Pandera."""
        logger.info("Validating with Pandera...")
        self.validator.validate(df)
        logger.info("Validation passed")
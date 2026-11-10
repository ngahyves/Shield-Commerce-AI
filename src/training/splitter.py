"""Temporal train/test split. Used before training, calibration, and evaluation."""

import pandas as pd

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

logger = ProjectLogger(__name__).get_logger()


class TemporalSplitter:
    """Split a DataFrame into train/test by transaction_dt."""

    def __init__(self) -> None:
        self.config = ConfigLoader()
        self.seconds_per_day = self.config.get("constants", "seconds_per_day", default=86400)
        self.test_start_day = self.config.get("split", "test_start_day")
        self.test_end_day = self.config.get("split", "test_end_day")

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train_df, test_df).

        Args:
            df: Full validated DataFrame.

        Returns:
            (train_df, test_df) split strictly by transaction_dt.
        """
        test_start_dt = self.test_start_day * self.seconds_per_day
        test_end_dt = self.test_end_day * self.seconds_per_day

        train = df[df["transaction_dt"] < test_start_dt].copy()
        test = df[(df["transaction_dt"] >= test_start_dt) &
                  (df["transaction_dt"] < test_end_dt)].copy()

        logger.info(
            "Temporal split — train: %d rows (%.3f%% fraud), test: %d rows (%.3f%% fraud)",
            len(train), 100 * train["is_fraud"].mean(),
            len(test), 100 * test["is_fraud"].mean(),
        )
        return train, test
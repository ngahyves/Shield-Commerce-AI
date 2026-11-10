"""One-time script: index confirmed training frauds into Qdrant."""

import sys
from pathlib import Path

from src.ingestion.ingest import GoldIngestor
from src.training.splitter import TemporalSplitter
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger
from src.vector.indexer import build_index, recreate_collection

log = ProjectLogger(__name__).get_logger()


def main() -> int:
    cfg = ConfigLoader()
    collection = cfg.get("qdrant", "collection", default="shieldcommerce_fraud_cases")
    models_dir = Path(cfg.get("paths", "models"))
    test_start_day = cfg.get("split", "test_start_day", default=155)
    seconds_per_day = cfg.get("constants", "seconds_per_day", default=86_400)

    df = GoldIngestor().ingest()
    train_df, _ = TemporalSplitter().split(df)

    recreate_collection(collection)
    n = build_index(train_df, collection, test_start_day * seconds_per_day,
                    scaler_path=models_dir / "similarity_scaler.pkl")

    print(f"\nIndexed {n} fraud cases into '{collection}'")
    print(f"Scaler saved: {models_dir / 'similarity_scaler.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
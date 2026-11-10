"""Verify the TargetEncoderBuilder on real data.

Runs the full pipeline: load → split → fit → verify → save.
"""

import sys

from src.ingestion.ingest import GoldIngestor
from src.features.target_encoder import TargetEncoderBuilder
from src.training.splitter import TemporalSplitter


def main() -> int:
    # 1. Load + split (train only, test held out)
    df = GoldIngestor().ingest()
    train_df, test_df = TemporalSplitter().split(df)

    builder = TargetEncoderBuilder()

    # 2. Prepare X / y
    X_train = train_df.drop(columns=builder.metadata_cols)
    y_train = train_df["is_fraud"]
    X_test = test_df.drop(columns=builder.metadata_cols)

    # 3. Fit on train ONLY
    builder.fit(X_train, y_train)

    # 4. Verify no leakage
    print("\n" + "=" * 60)
    print("LEAKAGE VERIFICATION")
    print("=" * 60)
    diag = builder.verify_no_leakage(X_train, y_train, X_test)

    # 5. Transform train + test
    X_train_enc = builder.transform(X_train)
    X_test_enc = builder.transform(X_test)

    print("\n" + "=" * 60)
    print("SHAPES")
    print("=" * 60)
    print(f"X_train raw:        {X_train.shape}")
    print(f"X_train encoded:    {X_train_enc.shape}")
    print(f"X_test raw:         {X_test.shape}")
    print(f"X_test encoded:     {X_test_enc.shape}")
    print(f"Output columns:     {list(X_train_enc.columns)[:10]}...")

    # 6. Save
    path = builder.save()
    print(f"\nSaved to: {path}")

    return 0 if diag["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
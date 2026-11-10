"""Drift analysis: train vs test set using Evidently (v1.0+ API)."""

from datetime import datetime
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from src.agents.tools import _gold
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def main() -> None:
    cfg = ConfigLoader()
    metadata_cols = cfg.get("features", "metadata_cols")
    seconds_per_day = cfg.get("constants", "seconds_per_day", default=86400)
    train_start_day = cfg.get("split", "train_start_day", default=1)
    test_start_day = cfg.get("split", "test_start_day")
    test_end_day = cfg.get("split", "test_end_day")
    drift_threshold = cfg.get("monitoring", "drift_threshold", default=0.20)

    log.info("loading gold...")
    df = _gold()

    reference_start_dt = train_start_day * seconds_per_day
    test_start_dt = test_start_day * seconds_per_day
    test_end_dt = test_end_day * seconds_per_day

    is_reference = (df["transaction_dt"] >= reference_start_dt) & (df["transaction_dt"] < test_start_dt)
    is_current = (df["transaction_dt"] >= test_start_dt) & (df["transaction_dt"] < test_end_dt)

    features = df.drop(columns=metadata_cols, errors="ignore")
    reference = features[is_reference].reset_index(drop=True)
    current = features[is_current].reset_index(drop=True)

    log.info("reference: %d rows | current: %d rows | features: %d",
             len(reference), len(current), len(features.columns))
    log.info("drift_share threshold: %.2f", drift_threshold)

    # NEW API: DataDriftPreset with num_method / cat_method
    report = Report([
        DataDriftPreset(
            drift_share=drift_threshold,
            num_method="psi",
            cat_method="chisquare",
            num_threshold=0.2,       # PSI ≥ 0.2 = drift
            cat_threshold=0.05,      # p-value < 0.05 = drift
        ),
    ])

    log.info("running drift report ...")
    evaluation = report.run(
        current_data=current,
        reference_data=reference,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = reports_dir / f"drift_train_vs_test_{ts}.html"
    evaluation.save_html(str(html_path))
    log.info("report saved: %s", html_path)

    # Console summary
    result = evaluation.dict()
    _print_summary(result)


def _print_summary(report_dict: dict) -> None:
    """Extract and log the drift summary from an Evidently Evaluation dict."""
    metrics = report_dict.get("metrics", [])

    # Dataset-level: DriftedColumnsCount
    dataset_share = None
    n_drifted = None
    for m in metrics:
        if m.get("metric") == "DriftedColumnsCount":
            res = m.get("result", {})
            n_drifted = int(res.get("count", 0))
            dataset_share = float(res.get("share", 0.0))
            break

    if dataset_share is not None:
        log.info("=" * 60)
        log.info("DRIFT SHARE      : %.1f%% (%d columns drifted)", dataset_share * 100, n_drifted)

    # Column-level: ValueDrift entries
    drifted = []
    for m in metrics:
        if m.get("metric") == "ValueDrift":
            res = m.get("result", {})
            col = res.get("column", "?")
            method = res.get("method", "?")
            score = res.get("drift_score", 0.0)
            detected = res.get("drift_detected", False)
            if detected:
                drifted.append({"col": col, "method": method, "score": float(score or 0.0)})

    drifted.sort(key=lambda x: -x["score"])
    log.info("TOP DRIFTED FEATURES (detected):")
    for d in drifted[:10]:
        log.info("  %-30s %-12s score=%.4f", d["col"], d["method"], d["score"])


if __name__ == "__main__":
    main()
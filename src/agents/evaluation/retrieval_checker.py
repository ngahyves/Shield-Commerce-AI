"""Deterministic retrieval checks (Q1) for the fraud agent evaluation.

Verifies that the 4 retrieval tools return correct data from the Gold Parquet. Every check
is a direct comparison against the source of truth.
"""

import json
import math
from pathlib import Path

from src.agents.tools import (_gold, get_customer, get_shap_explanations, get_transaction)
from src.agents.tools_similar import search_similar_cases
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def _norm(val):
    """Normalize a value to Python native, converting NaN to None."""
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


class RetrievalChecker:
    """Verifies that the retrieval tools return correct data from the Gold."""

    def __init__(self, config_path: str = "config.yaml"):
        cfg = ConfigLoader(config_path)
        self.models_dir = Path(cfg.get("paths", "models"))
        split_start = cfg.get("split", "test_start_day")
        spd = cfg.get("constants", "seconds_per_day", default=86400)
        self.train_end_dt = split_start * spd
        self._feature_names: set | None = None

    def _load_feature_names(self) -> set:
        if self._feature_names is None:
            with open(self.models_dir / "model_metadata.json") as f:
                meta = json.load(f)
            self._feature_names = set(meta["feature_names_out"])
        return self._feature_names

    def check_transaction(self, tx_id: int) -> dict:
        """Verify get_transaction returns the same values as the Gold row."""
        tx = get_transaction.invoke({"transaction_id": tx_id})
        df = _gold()
        if tx_id not in df.index:
            return {"pass": False, "issues": [f"tx {tx_id} not in Gold"]}
        gold_row = df.loc[[tx_id]]

        issues = []
        for field in ("amount", "hour_of_day", "uid"):
            if field not in gold_row.columns:
                issues.append(f"{field}: column missing from Gold")
                continue
            tx_val = _norm(tx.get(field))
            gold_val = _norm(gold_row[field].iloc[0])
            if tx_val != gold_val:
                issues.append(f"{field}: tool={tx_val} vs gold={gold_val}")

        return {"pass": len(issues) == 0, "issues": issues}

    def check_customer(self, tx_id: int) -> dict:
        """Verify get_customer aggregates match direct Gold computation."""
        tx = get_transaction.invoke({"transaction_id": tx_id})
        uid = tx["uid"]
        tx_dt = tx["transaction_dt"]

        cust = get_customer.invoke({"uid": uid, "transaction_dt": tx_dt})

        df = _gold()
        prior = df[(df["uid"] == uid) & (df["transaction_dt"] < tx_dt)]
        expected_n = len(prior)
        expected_new = expected_n == 0

        issues = []
        if cust["n_transactions"] != expected_n:
            issues.append(
                f"n_transactions: tool={cust['n_transactions']} vs gold={expected_n}"
            )
        if cust["is_new_customer"] != expected_new:
            issues.append(
                f"is_new_customer: tool={cust['is_new_customer']} vs gold={expected_new}"
            )

        return {"pass": len(issues) == 0, "issues": issues}

    def check_shap(self, tx_id: int) -> dict:
        """Verify SHAP factors: features valid, sorted, direction consistent."""
        factors = get_shap_explanations.invoke({"transaction_id": tx_id})
        feature_names = self._load_feature_names()

        issues = []
        if len(factors) != 5:
            issues.append(f"expected 5 factors, got {len(factors)}")

        for f in factors:
            if f["feature"] not in feature_names:
                issues.append(f"feature {f['feature']} not in feature_names_out")

        if factors:
            max_abs = max(abs(f["shap_value"]) for f in factors)
            if abs(factors[0]["shap_value"]) < max_abs - 1e-9:
                issues.append(
                    f"rank 1 ({factors[0]['feature']}) not max |shap| "
                    f"(|shap|={abs(factors[0]['shap_value']):.4f} vs {max_abs:.4f})"
                )

        for f in factors:
            expected = "positive" if f["shap_value"] >= 0 else "negative"
            if f["direction"] != expected:
                issues.append(
                    f"{f['feature']}: direction={f['direction']} "
                    f"but shap={f['shap_value']:.4f}"
                )

        return {"pass": len(issues) == 0, "issues": issues}

    def check_similar_cases(self, tx_id: int) -> dict:
        """Verify similar cases: 5 items, anti-leakage, score threshold."""
        tx = get_transaction.invoke({"transaction_id": tx_id})
        tx_dt = tx["transaction_dt"]

        cases = search_similar_cases.invoke({
            "query_row": tx,
            "transaction_dt": tx_dt,
        })

        issues = []
        if len(cases) != 5:
            issues.append(f"expected 5 cases, got {len(cases)}")

        for c in cases:
            case_dt = c["transaction_dt"]
            if case_dt >= self.train_end_dt:
                issues.append(
                    f"case {c['transaction_id']}: dt={case_dt} "
                    f">= train_end_dt={self.train_end_dt}"
                )
            if case_dt >= tx_dt:
                issues.append(
                    f"case {c['transaction_id']}: dt={case_dt} "
                    f">= current_dt={tx_dt} (leak)"
                )
            if c["score"] < 0.6:
                issues.append(
                    f"case {c['transaction_id']}: score={c['score']:.4f} < 0.6"
                )

        return {"pass": len(issues) == 0, "issues": issues}

    def run(self, tx_ids: list[int]) -> dict:
        """Run the 4 retrieval checks on every transaction."""
        log.info("Q1 retrieval: %d transactions", len(tx_ids))

        results = {
            "summary": {"total": 0, "pass": 0, "fail": 0, "pass_rate": 0.0},
            "per_check": {
                "transaction": {"pass": 0, "fail": 0},
                "customer": {"pass": 0, "fail": 0},
                "shap": {"pass": 0, "fail": 0},
                "similar_cases": {"pass": 0, "fail": 0},
            },
            "per_tx": {},
        }

        checks = [
            ("transaction", self.check_transaction),
            ("customer", self.check_customer),
            ("shap", self.check_shap),
            ("similar_cases", self.check_similar_cases),
        ]

        for tx_id in tx_ids:
            tx_results = {}
            for check_name, check_fn in checks:
                try:
                    result = check_fn(tx_id)
                except Exception as exc:
                    result = {"pass": False, "issues": [f"exception: {exc}"]}

                tx_results[check_name] = result
                results["summary"]["total"] += 1

                if result["pass"]:
                    results["summary"]["pass"] += 1
                    results["per_check"][check_name]["pass"] += 1
                else:
                    results["summary"]["fail"] += 1
                    results["per_check"][check_name]["fail"] += 1
                    log.warning(
                        "Q1 fail: tx=%d check=%s issues=%s",
                        tx_id, check_name, result["issues"],
                    )

            results["per_tx"][tx_id] = tx_results

        total = results["summary"]["total"]
        if total > 0:
            results["summary"]["pass_rate"] = results["summary"]["pass"] / total

        log.info(
            "Q1 retrieval: %d/%d pass (%.1f%%)",
            results["summary"]["pass"], total,
            results["summary"]["pass_rate"] * 100,
        )
        return results


if __name__ == "__main__":
    # Quick local test on 2 tx (1 fraud + 1 legit) from the test set
    df = _gold()
    test_df = df[df["transaction_dt"] >= 155 * 86400]
    frauds = test_df[test_df["is_fraud"] == 1].sample(1, random_state=42)
    legits = test_df[test_df["is_fraud"] == 0].sample(1, random_state=42)
    tx_ids = [
        int(frauds["transaction_id"].iloc[0]),
        int(legits["transaction_id"].iloc[0]),
    ]

    print(f"Testing on tx_ids: {tx_ids}")
    checker = RetrievalChecker()
    result = checker.run(tx_ids)
    print(json.dumps(result["summary"], indent=2))
    print("\nPer-check results:")
    print(json.dumps(result["per_check"], indent=2))
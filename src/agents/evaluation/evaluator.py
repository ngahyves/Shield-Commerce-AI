"""Orchestrator - runs Q1 (retrieval) + Q2 (hallucination).
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.agents.evaluation.hallucination_judge import HallucinationJudge
from src.agents.evaluation.retrieval_checker import RetrievalChecker
from src.agents.fraud_investigator import investigate_transaction, score_transaction
from src.agents.tools import _gold
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


class AgentEvaluator:
    """End-to-end evaluation of the agent: Q1 + Q2."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        cfg = ConfigLoader(config_path)
        spd = cfg.get("constants", "seconds_per_day", default=86400)
        self.test_start_dt = cfg.get("split", "test_start_day") * spd
        self.output_dir = Path("reports/evaluation")

    def _select_from_group(self, group_df, target: int, seed: int) -> list[int]:
        selected: list[int] = []
        for tx_id in group_df["transaction_id"].sample(frac=1.0, random_state=seed):
            if len(selected) >= target:
                break
            _, _, decision = score_transaction(int(tx_id))
            if decision == "review":
                selected.append(int(tx_id))
        return selected

    def _select_tx_ids(self, n_fraud: int = 5, n_legit: int = 5, seed: int = 42) -> list[int]:
        df = _gold()
        test_df = df[df["transaction_dt"] >= self.test_start_dt]
        frauds = test_df[test_df["is_fraud"] == 1]
        legits = test_df[test_df["is_fraud"] == 0]
        fraud_ids = self._select_from_group(frauds, n_fraud, seed)
        legit_ids = self._select_from_group(legits, n_legit, seed)
        if len(fraud_ids) < n_fraud or len(legit_ids) < n_legit:
            log.warning(
                "Only found %d frauds and %d legits in review zone",
                len(fraud_ids), len(legit_ids),
            )
        return fraud_ids + legit_ids

    def _run_explain(self, tx_ids: list[int]) -> dict[int, dict]:
        df = _gold()
        states: dict[int, dict] = {}
        for i, tx_id in enumerate(tx_ids):
            if i > 0:
                log.info("Sleeping 25s to respect Groq rate limit...")
                time.sleep(25)

            log.info("Running /explain on tx=%d (%d/%d)", tx_id, i + 1, len(tx_ids))
            try:
                state = investigate_transaction(tx_id)
                label = int(df.loc[[tx_id]]["is_fraud"].iloc[0])
                states[tx_id] = {"state": state, "label": label}
            except Exception as exc:
                log.error("Failed on tx=%d: %s", tx_id, exc)
        return states

    def run(self, n_fraud: int = 5, n_legit: int = 5) -> dict:
        log.info("=" * 60)
        log.info("Agent evaluation starting (Q1 + Q2)")
        log.info("=" * 60)

        tx_ids = self._select_tx_ids(n_fraud=n_fraud, n_legit=n_legit)
        log.info("Selected %d transactions: %s", len(tx_ids), tx_ids)
        if not tx_ids:
            raise RuntimeError("No transactions selected for evaluation")

        # Q1
        log.info("-" * 60)
        log.info("[Q1] Retrieval correctness (deterministic)")
        q1 = RetrievalChecker().run(tx_ids)

        # /explain
        log.info("-" * 60)
        log.info("[EXPLAIN] Running agent on each transaction")
        states = self._run_explain(tx_ids)

        # Q2
        log.info("-" * 60)
        log.info("[Q2] Hallucination detection")
        try:
            judge = HallucinationJudge()
            states_only = {tx_id: e["state"] for tx_id, e in states.items()}
            q2 = judge.run(states_only)
        except Exception as exc:
            log.error("Q2 skipped: %s", exc)
            q2 = {"mean_score": None, "n_judged": 0, "n_above_0_5": 0, "per_case": {}}

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {"n_fraud": n_fraud, "n_legit": n_legit},
            "tx_ids": tx_ids,
            "labels": {str(tx): e["label"] for tx, e in states.items()},
            "q1_retrieval": q1,
            "q2_hallucination": q2,
            "summary": {
                "q1_pass_rate": q1["summary"]["pass_rate"],
                "q2_mean_score": q2["mean_score"],
                "q2_cases_above_0_5": q2["n_above_0_5"],
            },
        }
        self._write_report(report)
        self._log_summary(report["summary"])
        return report

    def _write_report(self, report: dict) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"agent_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Report written: %s", path)
        return path

    def _log_summary(self, summary: dict) -> None:
        log.info("=" * 60)
        log.info("EVALUATION SUMMARY")
        log.info("=" * 60)
        log.info("Q1 retrieval pass rate : %.1f%%", summary["q1_pass_rate"] * 100)
        if summary["q2_mean_score"] is not None:
            log.info(
                "Q2 hallucination mean  : %.2f (%d case(s) > 0.5)",
                summary["q2_mean_score"], summary["q2_cases_above_0_5"],
            )
        else:
            log.info("Q2 hallucination: SKIPPED")


if __name__ == "__main__":
    evaluator = AgentEvaluator()
    evaluator.run(n_fraud=5, n_legit=5)
# src/agents/test_agent.py
from dotenv import load_dotenv
load_dotenv()  

from src.ingestion.ingest import GoldIngestor
from src.training.splitter import TemporalSplitter
from src.agents.retrieval import retrieval_node
from src.agents.investigator import investigator_node
from src.agents.reporter import reporter_node   

df = GoldIngestor().ingest()
_, test_df = TemporalSplitter().split(df)

row = test_df.iloc[100]
state = {
    "transaction_id": int(row["transaction_id"]),
    "model_decision": "review",
    "calibrated_proba": 0.15,
    "raw_score": 0.42,
    "model_version": "1.0.0",
}

state.update(retrieval_node(state))

print("=== INVESTIGATION ===")
state.update(investigator_node(state))
print(state.get("investigation", "(empty)"))

if state.get("errors"):
    print("\nERRORS:", state["errors"])

print("\n=== REPORT ===")
state.update(reporter_node(state))
report = state["report"]
print(f"recommendation: {report['recommendation']}")
print(f"confidence:     {report['confidence']}")
print(f"reasoning:      {report['reasoning']}")
print(f"evidence:       {len(report['evidence'])} items")
print(f"similar_cases:  {len(report['similar_cases'])}")
print(f"audit_id:       {report['audit_id'][:30]}...")
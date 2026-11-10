"""API-level schemas."""

from pydantic import BaseModel, Field

from src.agents.schemas import InvestigationReport

class PredictResponse(BaseModel):
    transaction_id: int
    raw_score: float
    calibrated_proba: float
    decision: str  # "approve" | "review" | "block"
    model_name: str       
    model_version: str
    thresholds: dict


class EvidenceBundle(BaseModel):
    """Structured evidence for UI rendering (charts, timelines)."""

    shap_factors: list[dict] = Field(default_factory=list)
    customer_profile: dict = Field(default_factory=dict)
    related_transactions: list[dict] = Field(default_factory=list)
    similar_cases: list[dict] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    """Full agentic output: machine + human + structured layers."""

    report: InvestigationReport
    investigation: str
    evidence_bundle: EvidenceBundle
    route: str
    latency_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
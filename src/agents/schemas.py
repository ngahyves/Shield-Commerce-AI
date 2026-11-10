"""Pydantic contracts for the fraud investigation agent.

Two families of models:
- Tool I/O: what each tool receives and returns (strict, validated).
- Report: the final InvestigationReport produced by the reporting agent.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# ── Tool inputs ──────────────────────────────────────────────────────

class GetTransactionInput(BaseModel):
    transaction_id: int = Field(..., description="Transaction identifier")


class GetRelatedTransactionsInput(BaseModel):
    uid: str = Field(..., description="Customer UID (card_id_addr1_d1n)")
    transaction_dt: int = Field(..., description="Current transaction time index (seconds)")
    hours: int = Field(24, ge=1, le=720, description="Look-back window in hours")
    limit: int = Field(10, ge=1, le=50, description="Number of related transactions")


class GetShapExplanationsInput(BaseModel):
    transaction_id: int
    top_n: int = Field(5, ge=1, le=20)


class GetCustomerInput(BaseModel):
    uid: str


class SearchSimilarCasesInput(BaseModel):
    query_text: str = Field(..., description="Textual description of the transaction")
    transaction_dt: int = Field(..., description="Current time index; results are filtered < this value")
    limit: int = Field(5, ge=1, le=20)


# ── Tool outputs ─────────────────────────────────────────────────────

class TransactionDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    uid: str
    transaction_dt: int
    amount: float
    product_type: str | None
    card_id: int | None
    card_bank_1: float | None
    buyer_region: str | None
    device_family: str | None
    hour_of_day: int | None
    velocity_count_1h: int | None
    velocity_count_24h: int | None
    uid_amount_mean: float | None
    uid_amount_std: float | None
    uid_transaction_count: int | None
    is_null_dist1: int | None
    # Ground truth from the Gold dataset. Available only because this is a
    # labeled offline dataset. In production the label arrives weeks later
    # via chargeback and would NOT be visible to the agent.
    is_fraud: int


class RelatedTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    transaction_dt: int
    amount: float
    hours_before: float
    is_fraud: int


class ShapFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    shap_value: float
    feature_value: float | None
    direction: Literal["positive", "negative"]


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    is_new_customer: bool = False
    first_seen_days_ago: float
    last_activity_days_ago: float
    n_transactions: int
    n_frauds_confirmed: int
    avg_amount: float
    std_amount: float
    max_amount: float


class SimilarCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    score: float
    transaction_dt: int
    amount: float
    fraud_type: str | None


# ── Final report ─────────────────────────────────────────────────────

class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(..., description="Factual claim stated in the report")
    source: str = Field(..., description="Tool that produced the evidence")
    field: str = Field(..., description="Field or derived value referenced")
    value: str | int | float | bool | None = None


class InvestigationReport(BaseModel):
    """Structured output of the reporting agent. This is the final artifact."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    recommendation: Literal["approve", "review", "block"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(..., min_length=20, max_length=1200)
    evidence: list[Evidence] = Field(..., min_length=1)

    # return [] and never fabricate similar cases when the tool is absent.
    similar_cases: list[SimilarCase] = Field(default_factory=list)

    suggested_actions: list[str] = Field(default_factory=list)

    # Model context: preserved in the report so the analyst can audit the
    # full decision chain (model → router → agent → recommendation).
    calibrated_proba: float
    raw_score: float
    model_decision: Literal["approve", "review", "block"]
    model_version: str

    latency_ms: dict[str, float] = Field(default_factory=dict)
    audit_id: str
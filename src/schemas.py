from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    # Policy / RAG intents
    RETURN_POLICY = "return_policy"
    REFUND_POLICY = "refund_policy"
    SHIPPING_POLICY = "shipping_policy"
    POLICY_UNKNOWN = "policy_unknown"        # ← NEW

    # Order tool intents
    ORDER_TRACKING = "order_tracking"
    CANCEL_ORDER = "cancel_order"

    # Escalation intents
    HUMAN_HANDOFF = "human_handoff"
    COMPLAINT = "complaint"
    AGENT_REQUEST = "agent_request"

    # Conversational intents
    GREETING = "greeting"
    THANKS = "thanks"
    CAPABILITIES = "capabilities"
    OUT_OF_SCOPE = "out_of_scope"


class RequestAnalysis(BaseModel):
    intent: Intent
    order_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confirmation: str | None = None
    multi_intent: bool = False     
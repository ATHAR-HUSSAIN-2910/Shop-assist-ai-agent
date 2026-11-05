from typing import TypedDict, Annotated, Any, Optional
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    # Conversation messages (append-only via reducer)
    messages: Annotated[list, add_messages]

    # Customer / order context
    customer_id: Optional[str]
    order_id: Optional[str]

    # Request analysis
    intent: Optional[str]
    confidence: Optional[float]

    # Routing
    route: Optional[str]
    tool_name: Optional[str]

    # Transient data for the current turn
    retrieved_docs: list
    tool_result: Optional[Any]

    # Escalation
    should_escalate: bool
    escalation_reason: Optional[str]

    # Pending destructive action awaiting user confirmation
    pending_action: Optional[str]
    pending_order_id: Optional[str]

    # Final response
    response: Optional[str]

    confirmation: Optional[str] 
    multi_intent: bool
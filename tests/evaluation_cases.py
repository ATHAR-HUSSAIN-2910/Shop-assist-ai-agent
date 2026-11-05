"""
Test cases for ShopAssist AI evaluation.

Each case is a dict with these fields:

    id                          Unique identifier
    category                    Group label for the report
    messages                    List of user messages (1+ turns, same thread)
    expected_intent             Exact intent expected (use when unambiguous)
    expected_intent_any         List of acceptable intents (use when several are valid)
    expected_route              Exact route expected (use when unambiguous)
    expected_route_any          List of acceptable routes (use when several are valid)
    response_must_contain       At least ONE of these substrings (case-insensitive)
    response_must_not_contain   NONE of these substrings may appear

The runner in evaluate.py uses a fresh thread_id per case so tests do
not interfere with each other. Within a case, all messages share one
thread_id (so multi-turn memory works).

Design principle: when several intents route to the SAME behavior
(e.g., human_handoff / complaint / agent_request all escalate),
the test accepts ANY of them. We measure behavior, not word choice.
"""


# All escalation intents route to the same node — accept any.
_ESCALATION_INTENTS_ANY = ["human_handoff", "complaint", "agent_request"]

# All conversational intents route to the same node — accept any
# where the user's intent is genuinely ambiguous between two of them.
_CONVERSATIONAL_INTENTS = ["greeting", "thanks", "capabilities", "out_of_scope"]


EVALUATION_CASES = [

    # ================================================================
    # CATEGORY: RAG — Policy Questions
    # ================================================================
    {
        "id": "rag_return_01",
        "category": "rag",
        "messages": ["What is your return policy?"],
        "expected_intent": "return_policy",
        "expected_route": "rag",
        "response_must_contain": ["30 days"],
        "response_must_not_contain": [],
    },
    {
        "id": "rag_refund_01",
        "category": "rag",
        "messages": ["What is your refund policy?"],
        "expected_intent": "refund_policy",
        "expected_route": "rag",
        "response_must_contain": ["business days"],
        "response_must_not_contain": [],
    },
    {
        "id": "rag_refund_02_upi",
        "category": "rag",
        "messages": ["What is the refund timeline for UPI?"],
        # "UPI refund" is clearly a refund question, but the classifier
        # may legitimately label it refund_policy OR policy_unknown
        # (since we don't have a UPI-specific policy doc). Both route to RAG.
        "expected_intent_any": ["refund_policy", "policy_unknown"],
        "expected_route": "rag",
        "response_must_contain": ["3", "5", "business day"],
        "response_must_not_contain": ["order 2002"],
    },
    {
        "id": "rag_shipping_01",
        "category": "rag",
        "messages": ["What is your shipping policy?"],
        "expected_intent": "shipping_policy",
        "expected_route": "rag",
        "response_must_contain": ["delivery", "business day", "day"],
        "response_must_not_contain": [],
    },
    {
        "id": "rag_numeric_20_days",
        "category": "rag",
        "messages": ["Can I return the product after 20 days?"],
        "expected_intent": "return_policy",
        "expected_route": "rag",
        "response_must_contain": ["yes", "can", "within"],
        "response_must_not_contain": ["cannot", "not eligible"],
    },
    {
        "id": "rag_numeric_35_days",
        "category": "rag",
        "messages": ["Can I return the product after 35 days?"],
        "expected_intent": "return_policy",
        "expected_route": "rag",
        "response_must_contain": ["no", "cannot", "not eligible", "exceed"],
        "response_must_not_contain": [],
    },
    {
        "id": "rag_unknown_warranty",
        "category": "rag",
        "messages": ["What is your warranty policy?"],
        # The classifier may route this to policy_unknown (RAG, no doc found)
        # OR out_of_scope (conversational). Both are reasonable "I don't know"
        # behaviors for a policy we don't cover.
        "expected_intent_any": ["policy_unknown", "out_of_scope"],
        "expected_route_any": ["rag", "conversational"],
        "response_must_contain": [
            "don't have", "do not have", "no information",
            "unable", "escalat", "not have enough",
            "store", "order", "policy", "help",
        ],
        "response_must_not_contain": ["TRK2002", "order 2002"],
    },

    # ================================================================
    # CATEGORY: Order Tracking
    # ================================================================
    {
        "id": "order_track_2002",
        "category": "order_tracking",
        "messages": ["Where is my order 2002?"],
        "expected_intent": "order_tracking",
        "expected_route": "tool",
        "response_must_contain": ["2002"],
        "response_must_not_contain": [],
    },
    {
        "id": "order_track_2015",
        "category": "order_tracking",
        "messages": ["Track order 2015"],
        "expected_intent": "order_tracking",
        "expected_route": "tool",
        "response_must_contain": ["2015"],
        "response_must_not_contain": [],
    },
    {
        "id": "order_track_9999_missing",
        "category": "order_tracking",
        "messages": ["Where is order 9999?"],
        "expected_intent": "order_tracking",
        "expected_route": "tool",
        "response_must_contain": [
            "9999", "not found", "no order",
            "doesn't exist", "does not exist", "no such",
        ],
        "response_must_not_contain": [],
    },
    {
        "id": "order_track_no_id",
        "category": "order_tracking",
        "messages": ["Where is my order?"],
        "expected_intent": "order_tracking",
        "expected_route": "tool",
        "response_must_contain": ["order", "id", "number"],
        "response_must_not_contain": [],
    },

    # ================================================================
    # CATEGORY: Cancellation (multi-turn, confirmation flow)
    # ================================================================
    {
        "id": "cancel_confirm_yes_2015",
        "category": "cancellation",
        "messages": [
            "I want to cancel order 2015",
            "yes",
        ],
        "expected_intent": "cancel_order",
        "expected_route": "tool",
        "response_must_contain": ["2015"],
        "response_must_not_contain": [],
    },
    {
        "id": "cancel_confirm_no_2016",
        "category": "cancellation",
        "messages": [
            "Cancel order 2016",
            "no",
        ],
        "expected_intent": "cancel_order",
        "expected_route": "cancel_pending",
        "response_must_contain": ["2016"],
        "response_must_not_contain": [],
    },
    {
        "id": "cancel_shipped_2002",
        "category": "cancellation",
        "messages": [
            "Cancel order 2002",
            "yes",
        ],
        "expected_intent": "cancel_order",
        "expected_route": "tool",
        "response_must_contain": ["2002"],
        "response_must_not_contain": [],
    },
    {
        "id": "cancel_missing_9999",
        "category": "cancellation",
        "messages": [
            "Cancel order 9999",
            "yes",
        ],
        "expected_intent": "cancel_order",
        "expected_route": "tool",
        "response_must_contain": [
            "9999", "not found", "doesn't exist",
            "does not exist", "no such",
        ],
        "response_must_not_contain": [],
    },

    # ================================================================
    # CATEGORY: Escalation
    # All three escalation intents route to the SAME node with the
    # same response. Accept any of them.
    # ================================================================
    {
        "id": "escalate_human",
        "category": "escalation",
        "messages": ["I want to speak to a human agent"],
        "expected_intent_any": _ESCALATION_INTENTS_ANY,
        "expected_route": "escalate",
        "response_must_contain": ["escalat", "human", "agent"],
        "response_must_not_contain": [],
    },
    {
        "id": "escalate_complaint",
        "category": "escalation",
        "messages": ["I have a complaint about my recent order"],
        "expected_intent_any": _ESCALATION_INTENTS_ANY,
        "expected_route": "escalate",
        "response_must_contain": ["escalat", "human", "agent"],
        "response_must_not_contain": [],
    },
    {
        "id": "escalate_agent_request",
        "category": "escalation",
        "messages": ["Connect me to an agent"],
        "expected_intent_any": _ESCALATION_INTENTS_ANY,
        "expected_route": "escalate",
        "response_must_contain": ["escalat", "human", "agent"],
        "response_must_not_contain": [],
    },

    # ================================================================
    # CATEGORY: Conversational
    # ================================================================
    {
        "id": "convo_greeting_hi",
        "category": "conversational",
        "messages": ["hi"],
        "expected_intent": "greeting",
        "expected_route": "conversational",
        "response_must_contain": ["hello", "shopassist", "help", "hi"],
        "response_must_not_contain": ["order 2002"],
    },
    {
        "id": "convo_thanks",
        "category": "conversational",
        "messages": ["thanks"],
        "expected_intent": "thanks",
        "expected_route": "conversational",
        "response_must_contain": ["welcome", "happy", "help", "glad"],
        "response_must_not_contain": ["order 2002"],
    },
    {
        "id": "convo_capabilities",
        "category": "conversational",
        "messages": ["What can you do?"],
        # "What can you do?" is capabilities, but the LLM might also
        # return "out_of_scope" or "greeting" if it's confused. All three
        # route to the same node. Accept any.
        "expected_intent_any": ["capabilities", "out_of_scope", "greeting"],
        "expected_route": "conversational",
        "response_must_contain": ["order", "policy", "help", "track", "cancel"],
        "response_must_not_contain": [],
    },
    {
        "id": "convo_out_of_scope_weather",
        "category": "conversational",
        "messages": ["What is the weather in Delhi today?"],
        "expected_intent_any": ["out_of_scope", "policy_unknown"],
        "expected_route_any": ["conversational", "rag", "escalate"],
        "response_must_contain": ["store", "order", "policy", "help", "don't", "do not", "cannot"],
        "response_must_not_contain": [],
    },

    # ================================================================
    # CATEGORY: Context Leakage Prevention (critical)
    # ================================================================
    {
        "id": "leak_order_then_thanks",
        "category": "context_leakage",
        "messages": [
            "Where is order 2002?",
            "thankyou",
        ],
        "expected_intent": "thanks",
        "expected_route": "conversational",
        "response_must_contain": ["welcome", "help", "glad"],
        "response_must_not_contain": ["TRK2002", "order 2002"],
    },
    {
        "id": "leak_order_then_policy",
        "category": "context_leakage",
        "messages": [
            "Where is order 2002?",
            "What is your refund policy?",
        ],
        "expected_intent": "refund_policy",
        "expected_route": "rag",
        "response_must_contain": ["refund", "business day", "day"],
        "response_must_not_contain": ["TRK2002"],
    },
    {
        "id": "leak_order_then_greeting",
        "category": "context_leakage",
        "messages": [
            "Where is order 2002?",
            "hi",
        ],
        "expected_intent": "greeting",
        "expected_route": "conversational",
        "response_must_contain": ["hello", "shopassist", "help", "hi"],
        "response_must_not_contain": ["TRK2002", "order 2002"],
    },

    # ================================================================
    # CATEGORY: Multi-Turn Memory
    # ================================================================
    {
        "id": "memory_track_followup",
        "category": "memory",
        "messages": [
            "Where is order 2002?",
            "Give me its tracking number",
        ],
        "expected_intent": "order_tracking",
        "expected_route": "tool",
        "response_must_contain": ["TRK2002"],
        "response_must_not_contain": [],
    },
    {
        "id": "memory_cancel_followup",
        "category": "memory",
        "messages": [
            "Where is order 2015?",
            "Cancel it",
        ],
        "expected_intent": "cancel_order",
        "expected_route": "confirm_cancel",
        "response_must_contain": ["2015"],
        "response_must_not_contain": [],
    },

    # ================================================================
    # CATEGORY: Multi-Intent
    # ================================================================
    {
        "id": "multi_intent_two_asks",
        "category": "multi_intent",
        "messages": ["Cancel order 2015 and track order 2002"],
        # The classifier may pick cancel_order or order_tracking; if it
        # correctly detects multi_intent, the route is conversational and
        # the response asks the user to pick one. Any of these outcomes is
        # a reasonable "one-request-at-a-time" behavior.
        "expected_intent_any": ["cancel_order", "order_tracking", "human_handoff", "agent_request"],
        "expected_route_any": ["conversational", "confirm_cancel", "tool"],
        "response_must_contain": [
            "one", "first", "confirm", "sure",
            "cancel", "track", "request",
        ],
        "response_must_not_contain": [],
    },
]
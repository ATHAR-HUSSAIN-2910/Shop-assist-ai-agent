import json
import logging

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.database.connection import DATABASE_URL
from src.rag.retriever import KnowledgeRetriever
from src.schemas import RequestAnalysis
from src.state import SupportState
from src.tools.order_tools import track_order, cancel_order


# ---------- Logging setup (Change #14) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("shopassist")


load_dotenv()

client = Groq()
retriever = KnowledgeRetriever()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


CONVERSATIONAL_INTENTS = {"greeting", "thanks", "capabilities", "out_of_scope"}
RAG_INTENTS = {"return_policy", "refund_policy", "shipping_policy", "policy_unknown"}
TOOL_INTENTS = {"order_tracking", "cancel_order"}
ESCALATION_INTENTS = {"human_handoff", "complaint", "agent_request"}


# ---------- Helpers ----------

def _format_history(messages, limit=6):
    """Format the last N messages as readable conversation history ."""
    recent = messages[-limit:]
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "Other"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
def _call_groq_analyzer(messages):
    """Groq call with retry logic."""
    return client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "request_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": [
                                "return_policy", "refund_policy", "shipping_policy","policy_unknown",
                                "order_tracking", "cancel_order",
                                "human_handoff", "complaint", "agent_request",
                                "greeting", "thanks", "capabilities", "out_of_scope"
                            ]
                        },
                        "order_id": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "confirmation": {"type": ["string", "null"]},
                        "multi_intent": {"type": "boolean"}
                    },
                    "required": [
                        "intent", "order_id", "confidence",
                        "confirmation", "multi_intent"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True
)
def _call_groq_response(prompt):
    """Groq response call with retry logic (Change #16)."""
    return llm.invoke(prompt)


# ---------- Nodes ----------

def analyze_request(state: SupportState):
    logger.info("Analyzing the request...")

    user_message = state["messages"][-1].content
        # ---------- Deterministic guard for short conversational inputs ----------
    # The LLM sometimes classifies short messages like "thanks" or "hi" as
    # order_tracking due to context leakage from the previous turn.
    # For these very common, unambiguous cases, bypass the LLM entirely.
    normalized = user_message.strip().lower().rstrip("!.,?")

    simple_thanks = {
        "thanks", "thankyou", "thank you", "thx", "ty",
        "thanks!", "thank you so much", "appreciate it",
        "got it", "ok thanks", "okay thanks", "great thanks"
    }
    simple_greetings = {
        "hi", "hello", "hey", "yo", "hola",
        "good morning", "good afternoon", "good evening"
    }

    if normalized in simple_thanks:
        logger.info("Short-circuit: classified as 'thanks'")
        return {
            "intent": "thanks",
            "order_id": None,
            "confidence": 1.0,
            "confirmation": None,
            "multi_intent": False,
        }

    if normalized in simple_greetings:
        logger.info("Short-circuit: classified as 'greeting'")
        return {
            "intent": "greeting",
            "order_id": None,
            "confidence": 1.0,
            "confirmation": None,
            "multi_intent": False,
        }
    # ---------- End guard ----------
    history = _format_history(state["messages"][:-1])

    pending_action = state.get("pending_action")
    pending_order_id = state.get("pending_order_id")

    confirmation_hint = ""
    if pending_action == "cancel_order" and pending_order_id:
        confirmation_hint = (
            f"\nIMPORTANT: There is a pending action awaiting confirmation: "
            f"cancel_order for order {pending_order_id}. "
            f"If the user confirms (yes/sure/ok/confirm/go ahead), "
            f"set intent='cancel_order' and confirmation='yes'. "
            f"If the user declines (no/cancel/stop/don't), "
            f"set intent='cancel_order' and confirmation='no'. "
            f"Otherwise ignore the pending action and classify the new request.\n"
        )

    system_prompt = f"""You are a customer support request classifier.

Classify the customer's message into exactly one of these intents:

POLICY intents (the user asks about a company policy):
- return_policy       → returns, exchanges, return windows
- refund_policy       → refund timing, refund methods, refund status
- shipping_policy     → delivery times, shipping charges, delays
- policy_unknown      → any other store policy (warranty, privacy,
                        terms of service, loyalty program, etc.)

ORDER intents (only when the CURRENT message is clearly about an order):
- order_tracking      → tracking, delivery status, "where is my order"
- cancel_order        → cancel, stop, or confirm cancelling an order

ESCALATION intents:
- human_handoff, complaint, agent_request

CONVERSATIONAL intents (short, standalone, no order context needed):
- greeting      → "hi", "hello", "hey", "good morning", "good evening"
- thanks        → "thanks", "thank you", "thankyou", "thx",
                   "appreciate it", "got it", "ok thanks"
- capabilities  → "what can you do", "help", "how can you help"
- out_of_scope  → non-store topics (weather, sports, coding, general knowledge)

CRITICAL RULES:
1. Classify based ONLY on the CURRENT message, not the previous topic.
   If the previous turn was about an order but the new message asks
   about a policy, classify it as a POLICY intent, NOT order_tracking.
2. Extract the order ID ONLY if it appears in the CURRENT message.
   Do NOT copy the order ID from the conversation history.
3. If the message asks about a policy that isn't return/refund/shipping,
   use `policy_unknown`. Never default to order_tracking.
4. Reserve "out_of_scope" for non-store topics (weather, sports, general
   knowledge). Store-related policy questions go to policy_unknown.
5. Return a confidence score between 0 and 1.
6. If the user is confirming or declining a pending action, set
   "confirmation" to "yes", "no", or null.
7. If the user is asking multiple distinct things in one message,
   set "multi_intent" to true.
8. Short messages like "thanks", "thankyou", "hi", "ok", "cool", "got it"
   are CONVERSATIONAL intents — NOT order_tracking, even if the previous
   turn was about an order. Use the current message, not the topic.
9. Never carry an order_id forward from history. If the current message
   does not contain an order number, set order_id to null.
10. Do NOT answer the customer's question. Only classify.
{confirmation_hint}"""

    messages_for_llm = [{"role": "system", "content": system_prompt}]
    if history:
        messages_for_llm.append({
            "role": "user",
            "content": f"Previous conversation:\n{history}"
        })
    messages_for_llm.append({
        "role": "user",
        "content": f"Current message: {user_message}"
    })

    try:
        response = _call_groq_analyzer(messages_for_llm)
        data = json.loads(response.choices[0].message.content)
        analysis = RequestAnalysis.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Analyzer failed: {e}", exc_info=True)
        return {
            "intent": "human_handoff",
            "confidence": 0.0,
            "route": "escalate",
            "should_escalate": True,
            "escalation_reason": "Failed to analyze the request.",
            "retrieved_docs": [],
            "tool_result": None,
            "confirmation": None,
            "multi_intent": False,
        }
    except Exception as e:
        logger.error(f"Unexpected analyzer error: {e}", exc_info=True)
        return {
            "intent": "human_handoff",
            "confidence": 0.0,
            "route": "escalate",
            "should_escalate": True,
            "escalation_reason": "Analyzer service unavailable.",
            "retrieved_docs": [],
            "tool_result": None,
            "confirmation": None,
            "multi_intent": False,
        }

    order_id = analysis.order_id or state.get("order_id")

    return {
        "intent": analysis.intent.value,
        "order_id": order_id,
        "confidence": analysis.confidence,
        "confirmation": analysis.confirmation,
        "multi_intent": analysis.multi_intent,
    }


def route_request(state: SupportState):
    logger.info("Routing the request...")

    # Change #1: always clear transient state at the start of a turn
    base = {
        "retrieved_docs": [],
        "tool_result": None,
        "should_escalate": False,
        "escalation_reason": None,
        "response": None,
    }

    intent = state["intent"]
    confidence = state.get("confidence")
    confirmation = state.get("confirmation")
    pending_action = state.get("pending_action")
    pending_order_id = state.get("pending_order_id")

    # Change #8: multi-intent detection
    if state.get("multi_intent"):
        return {
            **base,
            "route": "conversational",
            "response": (
                "I can only handle one request at a time. "
                "Could you please tell me which one you'd like help with first?"
            ),
            "pending_action": None,
            "pending_order_id": None,
        }

    # Change #2: confirmation flow for pending cancellation
    if pending_action == "cancel_order" and pending_order_id:
        if confirmation == "yes":
            return {
                **base,
                "route": "tool",
                "pending_action": None,
                "pending_order_id": None,
            }
        if confirmation == "no":
            return {
                **base,
                "route": "cancel_pending",
                "pending_action": None,
                "pending_order_id": None,
            }
        if intent != "cancel_order":
            pending_action = None
            pending_order_id = None

    # Low-confidence escalation (unchanged)
    if confidence is not None and confidence < 0.60:
        return {
            **base,
            "route": "escalate",
            "should_escalate": True,
            "escalation_reason": (
                "The request could not be classified with enough confidence."
            ),
            "pending_action": None,
            "pending_order_id": None,
        }

    if intent in RAG_INTENTS:
        return {
            **base,
            "route": "rag",
            "order_id": None, 
            "pending_action": None,
            "pending_order_id": None,
        }

    # Change #2: new cancellation request → ask for confirmation first
    if intent == "cancel_order":
        order_id = state.get("order_id")
        if not order_id:
            return {
                **base,
                "route": "conversational",
                "response": (
                    "Sure — could you please tell me the order ID "
                    "you'd like to cancel?"
                ),
                "pending_action": None,
                "pending_order_id": None,
            }
        return {
            **base,
            "route": "confirm_cancel",
            "pending_action": "cancel_order",
            "pending_order_id": order_id,
        }

    if intent == "order_tracking":
        return {
            **base,
            "route": "tool",
            "pending_action": None,
            "pending_order_id": None,
        }

    if intent in ESCALATION_INTENTS:
        return {
            **base,
            "route": "escalate",
            "should_escalate": True,
            "pending_action": None,
            "pending_order_id": None,
        }

    # Change #7: conversational intents
    if intent in CONVERSATIONAL_INTENTS:
        return {
            **base,
            "route": "conversational",
            "order_id": None,
            "pending_action": None,
            "pending_order_id": None,
        }

    return {
        **base,
        "route": "escalate",
        "should_escalate": True,
        "escalation_reason": "Unsupported request.",
        "pending_action": None,
        "pending_order_id": None,
    }


def decide_next_node(state: SupportState):
    route = state["route"]
    return {
        "rag": "retrieve_knowledge",
        "tool": "execute_tool",
        "escalate": "escalate",
        "conversational": "handle_conversational",
        "confirm_cancel": "request_cancel_confirmation",
        "cancel_pending": "cancel_pending",
    }.get(route, "escalate")


def handle_conversational(state: SupportState):
    logger.info("Handling conversational request...")

    if state.get("response"):
        return {"response": state["response"]}

    intent = state["intent"]

    responses = {
        "greeting": (
            "Hello! I'm ShopAssist. I can help you track orders, "
            "cancel orders, or answer questions about our return, refund, "
            "and shipping policies. How can I help you today?"
        ),
        "thanks": (
            "You're welcome! Let me know if there's anything else "
            "I can help with."
        ),
        "capabilities": (
            "I can help with: order tracking, order cancellation, "
            "return/refund/shipping policy questions, and connecting "
            "you to a human agent if needed."
        ),
        "out_of_scope": (
            "I'm sorry, I can only help with store-related questions "
            "like orders and policies. Is there anything else I can help with?"
        ),
    }

    return {"response": responses.get(intent, "How can I help you today?")}


def request_cancel_confirmation(state: SupportState):
    logger.info("Requesting cancellation confirmation...")

    order_id = state.get("pending_order_id")
    return {
        "response": (
            f"Are you sure you want to cancel order {order_id}? "
            f"Please reply 'yes' to confirm or 'no' to keep the order."
        )
    }


def cancel_pending(state: SupportState):
    logger.info("Cancellation declined by user...")

    order_id = state.get("order_id")
    return {
        "response": (
            f"No problem — I won't cancel order {order_id}. "
            f"Is there anything else I can help with?"
        )
    }


def handle_escalation(state: SupportState):
    logger.info("Handling human handoff...")

    reason = state.get("escalation_reason") or (
        "Customer requested human assistance."
    )

    return {
        "should_escalate": True,
        "escalation_reason": reason,
        "response": (
            "I understand. I'm escalating your request to a human "
            "support agent for further assistance."
        )
    }


def retrieve_knowledge(state: SupportState):
    logger.info("Retrieving relevant knowledge...")

    query = state["messages"][-1].content
    documents = retriever.retrieve(query)

    # Change #3: if no relevant documents, flag it so the LLM says "I don't know"
    if not documents:
        return {
            "retrieved_docs": [],
            "tool_result": {
                "error": "No relevant policy information was found."
            }
        }

    return {"retrieved_docs": documents}


def execute_tool(state: SupportState):
    logger.info("Executing tool...")

    intent = state["intent"]
    order_id = state.get("order_id")

    if not order_id:
        return {"tool_result": {"error": "No order ID was provided."}}

    if intent == "order_tracking":
        result = track_order(order_id)
        if result is None:
            return {"tool_result": {"error": f"Order {order_id} was not found."}}
        return {"tool_result": result}

    if intent == "cancel_order":
        result = cancel_order(order_id)
        return {"tool_result": result}

    return {"tool_result": {"error": "This tool is not supported."}}


def generate_response(state: SupportState):
    logger.info("Generating the response...")

    retrieved_docs = state.get("retrieved_docs", [])
    tool_result = state.get("tool_result")
    user_query = state["messages"][-1].content
    history = _format_history(state["messages"][:-1])

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    tool_context = ""
    if tool_result:
        tool_context = f"\nTool Result:\n-------------\n{tool_result}\n-------------\n"

    prompt = f"""You are ShopAssist, an AI customer support agent.

Answer the customer's question using ONLY the information explicitly
provided in the Knowledge Base and Tool Result below.

Previous conversation:
----------------
{history if history else "(none)"}
----------------

Knowledge Base:
----------------
{context if context else "(no documents retrieved)"}
----------------
{tool_context}

Customer Question:
{user_query}

Rules:
1. The Tool Result is the source of truth for order information.
2. Never invent or assume any order information.
3. If a cancellation succeeded (status "cancelled"), say so clearly.
4. If the order was already cancelled, say so clearly.
5. Never mention order information not present in the Tool Result.
6. Never invent policies, prices, dates, guarantees, procedures, links, or contacts.
7. If the Tool Result contains an error, clearly explain that error.
8. If the available information is insufficient, say you do not have enough information.
9. Do not add unnecessary suggestions or assumptions.
10. Answer directly, concisely, and politely.
11. When the customer asks about a specific number (days, price, quantity),
    compare it explicitly against the policy value and state the comparison.
    Example: "Since 20 days is within the 30-day window, yes — you can return it."
    Do NOT rephrase the customer's number as if it were the policy value.
"""

    try:
        response = _call_groq_response(prompt)
        content = response.content
    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)
        content = (
            "Sorry, I'm having trouble generating a response right now. "
            "Please try again or ask for a human agent."
        )

    return {"response": content}


# ---------- Graph construction ----------

builder = StateGraph(SupportState)

builder.add_node("analyze_request", analyze_request)
builder.add_node("route_request", route_request)
builder.add_node("retrieve_knowledge", retrieve_knowledge)
builder.add_node("execute_tool", execute_tool)
builder.add_node("escalate", handle_escalation)
builder.add_node("handle_conversational", handle_conversational)
builder.add_node("request_cancel_confirmation", request_cancel_confirmation)
builder.add_node("cancel_pending", cancel_pending)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "analyze_request")
builder.add_edge("analyze_request", "route_request")

builder.add_conditional_edges(
    "route_request",
    decide_next_node,
    {
        "retrieve_knowledge": "retrieve_knowledge",
        "execute_tool": "execute_tool",
        "escalate": "escalate",
        "handle_conversational": "handle_conversational",
        "request_cancel_confirmation": "request_cancel_confirmation",
        "cancel_pending": "cancel_pending",
    }
)

builder.add_edge("retrieve_knowledge", "generate_response")
builder.add_edge("execute_tool", "generate_response")
builder.add_edge("generate_response", END)
builder.add_edge("escalate", END)
builder.add_edge("handle_conversational", END)
builder.add_edge("request_cancel_confirmation", END)
builder.add_edge("cancel_pending", END)


# Keep the PostgreSQL checkpointer alive for the lifetime of the graph.
checkpointer_context = PostgresSaver.from_conn_string(DATABASE_URL)
checkpointer = checkpointer_context.__enter__()
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)
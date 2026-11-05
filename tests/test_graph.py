import uuid

from langchain_core.messages import HumanMessage

from src.graph import graph


def run_test(name: str, message: str, expected_intent: str):
    """
    Helper: run one message through the graph and assert the intent.
    Uses a fresh thread_id each time so tests are independent.
    """
    thread_id = f"test-{uuid.uuid4()}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state = {
        "messages": [HumanMessage(content=message)]
    }

    result = graph.invoke(initial_state, config=config)

    actual_intent = result.get("intent")
    actual_route = result.get("route")
    actual_response = result.get("response", "")

    print(f"\n[TEST] {name}")
    print(f"  Message      : {message}")
    print(f"  Intent       : {actual_intent} (expected: {expected_intent})")
    print(f"  Route        : {actual_route}")
    print(f"  Response     : {actual_response[:80]}...")

    assert actual_intent == expected_intent, (
        f"FAIL: expected intent '{expected_intent}', got '{actual_intent}'"
    )
    assert actual_response, "FAIL: response is empty"

    print(f"  [PASS]")


def run_test_followup():
    """
    Test that the order ID is remembered across two turns
    in the same thread (conversation memory).
    """
    thread_id = f"followup-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1: mention order 2002
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="Where is my order 2002?")]},
        config=config
    )

    print(f"\n[TEST] Follow-up memory")
    print(f"  Turn 1 intent : {result1.get('intent')}")
    print(f"  Turn 1 order  : {result1.get('order_id')}")

    assert result1.get("order_id") == "2002", (
        f"FAIL: expected order_id '2002', got '{result1.get('order_id')}'"
    )

    # Turn 2: follow-up without mentioning the order ID
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Give me its tracking number.")]},
        config=config
    )

    print(f"  Turn 2 intent : {result2.get('intent')}")
    print(f"  Turn 2 order  : {result2.get('order_id')}")

    assert result2.get("order_id") == "2002", (
        f"FAIL: order_id was lost. Got '{result2.get('order_id')}'"
    )

    print(f"  [PASS]")


if __name__ == "__main__":
    run_test(
        name="Refund policy (RAG)",
        message="What is your refund policy?",
        expected_intent="refund_policy"
    )

    run_test(
        name="Order tracking (tool)",
        message="Where is my order 2002?",
        expected_intent="order_tracking"
    )

    run_test(
        name="Human handoff (escalation)",
        message="I want to speak to a human agent.",
        expected_intent="human_handoff"
    )

    run_test(
        name="Greeting (conversational)",
        message="Hi",
        expected_intent="greeting"
    )

    run_test_followup()

    print("\n===================================")
    print("ALL TESTS PASSED")
    print("===================================")
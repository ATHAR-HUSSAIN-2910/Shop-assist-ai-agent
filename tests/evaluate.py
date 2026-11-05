"""
Evaluation runner for ShopAssist AI.

Runs every case in tests/evaluation_cases.py through the LangGraph agent
and produces a pass/fail report grouped by category.

Run:
    python -m tests.evaluate

Design notes:
- Each test case gets a FRESH thread_id, so cases don't interfere.
- Multi-turn cases share ONE thread_id across all their messages.
- We check three things per case:
    1. intent  (with optional flexible matching)
    2. route   (with optional flexible matching)
    3. response keyword constraints
- A case passes only if ALL three checks pass.
"""

import sys
import uuid
import traceback

from langchain_core.messages import HumanMessage

from src.graph import graph
from tests.evaluation_cases import EVALUATION_CASES


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _normalize(text):
    """
    Normalize Unicode punctuation to ASCII so keyword matching is stable.

    The LLM often emits typographic quotes/apostrophes (’ ‘ “ ”) which
    would not match keyword strings written with ASCII ( ' " ).
    """
    if not text:
        return ""
    return (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")   # en dash
        .replace("—", "-")   # em dash
        .lower() 
    )


def run_case(case):
    """
    Execute a single test case and return the final state.
    Multi-turn cases send each message on the SAME thread.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = None
    for user_msg in case["messages"]:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )

    return result


def check_case(case, result):
    """
    Compare the graph result against the expected values.
    Returns a dict with detailed check outcomes.
    """
    actual_intent = result.get("intent")
    actual_route = result.get("route")
    actual_response = result.get("response") or ""
    response_lower = _normalize(actual_response)

    # Intent check — supports "expected_intent" or "expected_intent_any"
    expected_intent = case.get("expected_intent")
    expected_intent_any = case.get("expected_intent_any")

    if expected_intent_any:
        intent_ok = actual_intent in expected_intent_any
    else:
        intent_ok = actual_intent == expected_intent

    # Route check — supports "expected_route" or "expected_route_any"
    expected_route = case.get("expected_route")
    expected_route_any = case.get("expected_route_any")

    if expected_route_any:
        route_ok = actual_route in expected_route_any
    else:
        route_ok = actual_route == expected_route

    # Response keyword checks — normalize both sides so Unicode quotes
    # (from the LLM) and ASCII quotes (in test cases) match.
    must_contain = case.get("response_must_contain", [])
    must_not_contain = case.get("response_must_not_contain", [])

    contains_ok = (
        True
        if not must_contain
        else any(_normalize(kw) in response_lower for kw in must_contain)
    )
    not_contains_ok = (
        True
        if not must_not_contain
        else not any(_normalize(kw) in response_lower for kw in must_not_contain)
    )

    passed = intent_ok and route_ok and contains_ok and not_contains_ok

    return {
        "passed": passed,
        "intent_ok": intent_ok,
        "route_ok": route_ok,
        "contains_ok": contains_ok,
        "not_contains_ok": not_contains_ok,
        "actual_intent": actual_intent,
        "actual_route": actual_route,
        "actual_response": actual_response,
        "expected_intent": expected_intent,
        "expected_route": expected_route,
        "must_contain": must_contain,
        "must_not_contain": must_not_contain,
    }


def print_failure(case, outcome):
    print(f"\n[FAIL] {case['id']}  ({case['category']})")
    print(f"  Messages: {case['messages']}")

    if not outcome["intent_ok"]:
        print(f"  Intent    expected={outcome['expected_intent']!r}  got={outcome['actual_intent']!r}")
    if not outcome["route_ok"]:
        print(f"  Route     expected={outcome['expected_route']!r}   got={outcome['actual_route']!r}")
    if not outcome["contains_ok"]:
        print(f"  Response  must contain one of {outcome['must_contain']}")
    if not outcome["not_contains_ok"]:
        print(f"  Response  must NOT contain any of {outcome['must_not_contain']}")

    preview = outcome["actual_response"][:200].replace("\n", " ")
    print(f"  Response: {preview}...")


# ----------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------

def main():
    print("=" * 70)
    print("SHOPASSIST AI — EVALUATION REPORT")
    print("=" * 70)

    results_by_category = {}
    failures = []
    total_passed = 0
    total_cases = len(EVALUATION_CASES)

    for case in EVALUATION_CASES:
        category = case["category"]
        results_by_category.setdefault(category, {"passed": 0, "total": 0})
        results_by_category[category]["total"] += 1

        try:
            result = run_case(case)
            outcome = check_case(case, result)
        except Exception as e:
            # A crash is a failure; capture the traceback for debugging
            outcome = {
                "passed": False,
                "intent_ok": False,
                "route_ok": False,
                "contains_ok": False,
                "not_contains_ok": False,
                "actual_intent": None,
                "actual_route": None,
                "actual_response": f"EXCEPTION: {e}",
                "expected_intent": case.get("expected_intent"),
                "expected_route": case.get("expected_route"),
                "must_contain": case.get("response_must_contain", []),
                "must_not_contain": case.get("response_must_not_contain", []),
            }
            print(f"\n[CRASH] {case['id']}:")
            traceback.print_exc()

        if outcome["passed"]:
            results_by_category[category]["passed"] += 1
            total_passed += 1
        else:
            failures.append((case, outcome))

    # ---------- Summary table ----------
    print("\n" + "-" * 70)
    print(f"{'CATEGORY':<22} {'PASSED':>8} / {'TOTAL':<8}")
    print("-" * 70)

    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        print(f"{category:<22} {stats['passed']:>8} / {stats['total']:<8}")

    print("-" * 70)
    pct = (total_passed / total_cases * 100) if total_cases else 0
    print(f"{'TOTAL':<22} {total_passed:>8} / {total_cases:<8}   ({pct:.1f}%)")
    print("=" * 70)

    # ---------- Failure details ----------
    if failures:
        print(f"\nFAILED CASES ({len(failures)}):")
        print("-" * 70)
        for case, outcome in failures:
            print_failure(case, outcome)
        print("\n" + "=" * 70)

    # Exit code: 0 if all pass, 1 if any fail (useful for CI later)
    return 0 if total_passed == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
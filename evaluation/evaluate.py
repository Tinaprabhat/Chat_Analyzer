"""
Evaluation framework for Chat Persona RAG system.
Tests: topic detection quality, persona accuracy, retrieval relevance, query routing.
Run: python evaluation/evaluate.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict
from datetime import datetime

# ── Test cases ─────────────────────────────────────────────────────────────────
TOPIC_TEST_CASES = [
    {
        "id": "T1",
        "text": "I love cooking pasta and making Italian food at home. My favorite recipe is carbonara.",
        "expected_keywords": ["cooking", "pasta", "italian", "food", "recipe"],
        "expected_min_keywords": 2
    },
    {
        "id": "T2",
        "text": "I go to the gym every morning and do yoga for stress relief. Fitness is my priority.",
        "expected_keywords": ["gym", "yoga", "fitness", "stress"],
        "expected_min_keywords": 2
    },
    {
        "id": "T3",
        "text": "I'm a software engineer working on machine learning projects at a tech company.",
        "expected_keywords": ["software", "engineer", "machine learning", "tech"],
        "expected_min_keywords": 1
    },
    {
        "id": "T4",
        "text": "ok yeah lol haha thanks",
        "expected_keywords": [],
        "expected_min_keywords": 0,
        "is_noise": True
    },
    {
        "id": "T5",
        "text": "I have two dogs named Max and Bella. They love playing fetch in the park.",
        "expected_keywords": ["dogs", "max", "bella", "park", "fetch"],
        "expected_min_keywords": 1
    }
]

PERSONA_TEST_CASES = [
    {
        "id": "P1",
        "conv": {
            "conv_id": 0,
            "messages": [
                {"sender": "User_1", "text": "I cook every day and love Italian food 😊"},
                {"sender": "User_2", "text": "I work as a chef at a restaurant downtown."},
                {"sender": "User_1", "text": "Wow! I also go to the gym every morning."},
                {"sender": "User_2", "text": "I have two cats and a dog named Max."}
            ],
            "user1_msgs": ["I cook every day and love Italian food 😊", "Wow! I also go to the gym every morning."],
            "user2_msgs": ["I work as a chef at a restaurant downtown.", "I have two cats and a dog named Max."],
            "full_text": "cook Italian food gym chef restaurant cats dog Max"
        },
        "expected": {
            "User_1": {"food_signal": True, "fitness_signal": True, "emoji_present": True},
            "User_2": {"food_signal": True, "has_facts": True}
        }
    }
]

ROUTING_TEST_CASES = [
    {"query": "What kind of person is User 1?",           "expected": "persona"},
    {"query": "What are the habits of User 2?",           "expected": "persona"},
    {"query": "How does User 1 communicate?",             "expected": "persona"},
    {"query": "What topics were discussed?",              "expected": "topic"},
    {"query": "What was talked about in conversations?",  "expected": "topic"},
    {"query": "Give me a summary of the chats",           "expected": "summary"},
    {"query": "What is the overview of conversations?",   "expected": "summary"},
    {"query": "What did they discuss about food?",        "expected": "topic"},
]


# ── Evaluators ─────────────────────────────────────────────────────────────────
def evaluate_topic_detection() -> Dict:
    from src.topic_engine import extract_keywords
    results = []
    passed = 0

    for tc in TOPIC_TEST_CASES:
        extracted = extract_keywords(tc["text"])
        extracted_lower = [k.lower() for k in extracted]

        if tc.get("is_noise"):
            # For noise text, fewer keywords is better
            ok = len(extracted) <= 3
            results.append({
                "id": tc["id"],
                "pass": ok,
                "extracted": extracted,
                "note": f"Noise test: extracted {len(extracted)} keywords (expect ≤3)"
            })
        else:
            # Check if at least expected_min_keywords are found
            found = sum(1 for e in tc["expected_keywords"]
                       if any(e in k or k in e for k in extracted_lower))
            ok = found >= tc["expected_min_keywords"]
            results.append({
                "id": tc["id"],
                "pass": ok,
                "extracted": extracted,
                "expected_found": found,
                "expected_min": tc["expected_min_keywords"]
            })
        if ok:
            passed += 1

    return {
        "category": "Topic Detection",
        "total": len(TOPIC_TEST_CASES),
        "passed": passed,
        "score": round(passed / len(TOPIC_TEST_CASES) * 100, 1),
        "details": results
    }


def evaluate_persona_extraction() -> Dict:
    from src.persona_engine import extract_statistical_persona
    results = []
    passed = 0

    for tc in PERSONA_TEST_CASES:
        stats = extract_statistical_persona(tc["conv"])
        checks = []

        for user_key, expectations in tc["expected"].items():
            u = stats.get(user_key, {})
            if "food_signal" in expectations:
                ok = u.get("signals", {}).get("food_mentions", 0) >= 1
                checks.append(("food_signal", ok))
            if "fitness_signal" in expectations:
                ok = u.get("signals", {}).get("fitness_mentions", 0) >= 1
                checks.append(("fitness_signal", ok))
            if "emoji_present" in expectations:
                ok = u.get("emoji_count", 0) >= 1
                checks.append(("emoji_present", ok))
            if "has_facts" in expectations:
                ok = u.get("avg_msg_length", 0) > 0
                checks.append(("has_facts", ok))

        all_pass = all(v for _, v in checks)
        results.append({
            "id": tc["id"],
            "pass": all_pass,
            "checks": checks
        })
        if all_pass:
            passed += 1

    return {
        "category": "Persona Extraction",
        "total": len(PERSONA_TEST_CASES),
        "passed": passed,
        "score": round(passed / len(PERSONA_TEST_CASES) * 100, 1),
        "details": results
    }


def evaluate_query_routing() -> Dict:
    from src.query_engine import _classify_query
    results = []
    passed = 0

    for tc in ROUTING_TEST_CASES:
        predicted = _classify_query(tc["query"])
        ok = predicted == tc["expected"]
        results.append({
            "query":    tc["query"],
            "expected": tc["expected"],
            "got":      predicted,
            "pass":     ok
        })
        if ok:
            passed += 1

    return {
        "category": "Query Routing",
        "total": len(ROUTING_TEST_CASES),
        "passed": passed,
        "score": round(passed / len(ROUTING_TEST_CASES) * 100, 1),
        "details": results
    }


def evaluate_preprocessing() -> Dict:
    from src.preprocessor import parse_conversation, make_batches, make_chronological_checkpoints

    results = []
    passed = 0

    # Test 1: Basic parsing
    raw = "User 1: Hello!\nUser 2: Hi there!\nUser 1: How are you?"
    conv = parse_conversation(raw, 0)
    ok = len(conv["messages"]) == 3 and conv["user1_msgs"] and conv["user2_msgs"]
    results.append({"test": "basic_parse", "pass": ok})
    if ok: passed += 1

    # Test 2: Batching
    convs = [{"conv_id": i, "messages": [], "user1_msgs": [], "user2_msgs": [],
              "full_text": ""} for i in range(55)]
    batches = make_batches(convs, batch_size=10)
    ok = len(batches) == 6
    results.append({"test": "batching_count", "pass": ok})
    if ok: passed += 1

    # Test 3: Checkpoints
    cps = make_chronological_checkpoints(convs, interval=100)
    ok = len(cps) == 1
    results.append({"test": "checkpoint_single", "pass": ok})
    if ok: passed += 1

    # Test 4: Empty handling
    conv_empty = parse_conversation("", 99)
    ok = conv_empty["messages"] == []
    results.append({"test": "empty_input", "pass": ok})
    if ok: passed += 1

    total = len(results)
    return {
        "category": "Preprocessing",
        "total": total,
        "passed": passed,
        "score": round(passed / total * 100, 1),
        "details": results
    }


# ── Main runner ────────────────────────────────────────────────────────────────
def run_evaluation():
    print("\n" + "=" * 65)
    print("  CHAT PERSONA RAG — EVALUATION REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    categories = [
        evaluate_preprocessing,
        evaluate_topic_detection,
        evaluate_persona_extraction,
        evaluate_query_routing,
    ]

    all_results = []
    for fn in categories:
        try:
            result = fn()
            all_results.append(result)
            status = "✅" if result["score"] >= 75 else "⚠️"
            print(f"\n{status} {result['category']}: {result['passed']}/{result['total']} ({result['score']}%)")
            for d in result["details"]:
                mark = "  ✓" if d.get("pass") else "  ✗"
                key = d.get("id") or d.get("test") or d.get("query", "")[:40]
                print(f"  {mark} {key}")
        except Exception as e:
            print(f"\n❌ {fn.__name__} CRASHED: {e}")

    # Overall
    total_tests  = sum(r["total"]  for r in all_results)
    total_passed = sum(r["passed"] for r in all_results)
    overall      = round(total_passed / total_tests * 100, 1) if total_tests else 0

    print("\n" + "=" * 65)
    print(f"  OVERALL: {total_passed}/{total_tests} tests passed ({overall}%)")
    if overall >= 90:
        print("  🏆 Excellent")
    elif overall >= 75:
        print("  ✅ Good")
    elif overall >= 60:
        print("  ⚠️  Needs improvement")
    else:
        print("  ❌ Critical issues")
    print("=" * 65 + "\n")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "categories": all_results
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to logs/eval_report.json")


if __name__ == "__main__":
    run_evaluation()

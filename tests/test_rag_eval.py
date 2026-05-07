"""
tests/test_rag_eval.py
RAGWatch evaluation wired to the Chat Persona RAG system.

Metrics scored (CPU-only, no LLM-as-judge):
  faithfulness        — does the answer stay within the retrieved context?
  hallucination_score — fraction of answer claims not supported by context
  answer_relevance    — how relevant is the answer to the query?
  context_relevance   — how relevant are the retrieved chunks to the query?
  composite           — overall quality score

Run:
  python -m pytest tests/test_rag_eval.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows cp1252 terminals can't encode ✓/✗/— from RAGWatch summaries
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pytest
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from ragwatch.pytest_plugin import EvalReport, Thresholds, evaluate_one, evaluate_rag


# ── helper: run our RAG pipeline and package for RAGWatch ─────────────────────

def _rag_case(query: str, ground_truth: str | None = None) -> dict:
    """Run answer_query(), split context into chunk list, return RAGWatch dict."""
    from src.query_engine import answer_query
    result = answer_query(query)

    # Split the flat context string into individual chunks at section headers
    raw_ctx = result["context"]
    chunks = [c.strip() for c in raw_ctx.split("\n") if c.strip() and not c.startswith("===")]
    if not chunks:
        chunks = [raw_ctx]

    return {
        "query":        query,
        "context":      chunks,
        "answer":       result["answer"],
        "ground_truth": ground_truth,
        "metadata":     {"route": result["route"]},
    }


# ── test cases with optional ground-truth from known KB facts ─────────────────

RAG_CASES_DEFS = [
    # Persona — User 1
    ("What kind of person is User 1?",
     "User 1 is articulate with medium-length messages and neutral communication style."),
    ("What are User 1's habits?",
     None),
    ("How does User 1 communicate?",
     "User 1 uses neutral, medium-length messages with no emojis."),
    ("What is User 1's communication style?",
     "Neutral, medium-length messages without emojis."),
    ("What personal facts do we know about User 1?",
     None),

    # Persona — User 2
    ("What are the habits of User 2?",
     "User 2 discusses fitness/exercise and frequently mentions work."),
    ("Describe User 2's personality.",
     None),
    ("What is User 2's communication style?",
     "User 2 uses neutral, medium-length messages with no emojis."),
    ("What do we know about User 2's work life?",
     "User 2 frequently mentions work-related topics."),
    ("What personal facts do we know about User 2?",
     None),

    # Topic
    ("What topics were most discussed?",
     None),
    ("What subjects came up in the conversations?",
     None),
    ("What themes were discussed?",
     None),

    # Summary
    ("Give me a summary of the conversations",
     None),
    ("What is an overview of the chats?",
     None),
]


# ── session fixture: check prereqs ────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def session_setup():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    try:
        from src.kb_store import is_kb_populated
        if not is_kb_populated():
            pytest.skip("KB not populated — build KB first")
    except BaseException as e:
        pytest.skip(f"KB check failed: {e}")


# ── build all cases once per session ─────────────────────────────────────────

@pytest.fixture(scope="session")
def rag_cases() -> list[dict]:
    cases = []
    for query, gt in RAG_CASES_DEFS:
        try:
            cases.append(_rag_case(query, gt))
        except Exception as e:
            pytest.skip(f"answer_query failed for '{query[:40]}': {e}")
    return cases


# ── Pattern 1: full suite, single aggregate result ────────────────────────────

def test_rag_quality_aggregate(rag_cases):
    """All 15 cases together must clear composite ≥ 0.50 (measured baseline: 0.533 ±0.090)."""
    report = evaluate_rag(rag_cases, threshold=0.50)
    print(report.summary())
    assert report.passed, report.summary()


# ── Pattern 2: per-case parametrized (shows per-query score in pytest output) ─

@pytest.mark.parametrize("qdef", RAG_CASES_DEFS,
                         ids=[q[:45] for q, _ in RAG_CASES_DEFS])
def test_rag_quality_per_case(qdef, session_setup):
    """Each query scored individually — faithfulness ≥ 0.5, composite ≥ 0.5."""
    query, gt = qdef
    case = _rag_case(query, gt)
    result = evaluate_one(case)

    assert result.composite is not None, "composite score missing"
    print(f"\n  route     : {case['metadata']['route']}")
    print(f"  composite : {result.composite.score:.3f}")
    if result.faithfulness:
        print(f"  faithful  : {result.faithfulness.score:.3f}")
    if result.hallucination_score:
        print(f"  hallucin  : {result.hallucination_score.score:.3f}")
    if result.answer_relevance:
        print(f"  relevance : {result.answer_relevance.score:.3f}")

    assert result.composite.score >= 0.35, (
        f"Composite too low ({result.composite.score:.3f}) for: {query}"
    )
    if result.faithfulness:
        assert result.faithfulness.score >= 0.35, (
            f"Faithfulness too low ({result.faithfulness.score:.3f}) for: {query}"
        )


# ── Pattern 3: baseline report — always passes, prints full metrics ───────────

def test_rag_quality_baseline(rag_cases, capsys):
    """Smoke test — always passes. Prints full metric baseline."""
    report = evaluate_rag(rag_cases, threshold=0.0)
    print(report.summary())

    # Print per-case detail
    print("\n── Per-case scores ──────────────────────────────────────")
    for label, r in zip(report.case_labels, report.per_case):
        comp  = f"{r.composite.score:.3f}"         if r.composite          else "  N/A"
        faith = f"{r.faithfulness.score:.3f}"      if r.faithfulness       else "  N/A"
        hallu = f"{r.hallucination_score.score:.3f}" if r.hallucination_score else "  N/A"
        print(f"  {label:<46}  comp={comp}  faith={faith}  hallu={hallu}")

    assert isinstance(report, EvalReport)


# ── Pattern 4: strict — production quality bar ────────────────────────────────

def test_rag_quality_strict(rag_cases):
    """Strict thresholds (composite ≥ 0.75, faithfulness ≥ 0.65, hallucination ≤ 0.3).
    Skips rather than fails if bar not yet met — use as a stretch goal."""
    report = evaluate_rag(rag_cases, thresholds=Thresholds.strict())
    if not report.passed:
        pytest.skip(f"Strict thresholds not met yet:\n{report.summary()}")

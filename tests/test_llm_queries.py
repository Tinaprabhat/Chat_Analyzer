"""
tests/test_llm_queries.py
50-query integration test suite for the LLM query-answer pipeline.

- Auto-builds KB if CSV/PDF data exists but KB is empty.
- Skips gracefully if no data and no LLM available.
- Tests Groq primary → Ollama fallback chain.

Run:
  python -m pytest tests/test_llm_queries.py -v
  python -m pytest tests/test_llm_queries.py -v -k "persona"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── 50 query cases: (id, query, expected_route) ───────────────────────────────
QUERY_CASES = [
    # ── Persona: User 1 (1-15) ────────────────────────────────────────────────
    ( 1, "What kind of person is User 1?",               "persona"),
    ( 2, "What are User 1's habits?",                    "persona"),
    ( 3, "How does User 1 communicate?",                 "persona"),
    ( 4, "Describe User 1's personality",                "persona"),
    ( 5, "What is User 1's communication style?",        "persona"),
    ( 6, "Does User 1 use emojis?",                      "persona"),
    ( 7, "Is User 1 talkative or brief in messages?",    "persona"),
    ( 8, "What personal facts do we know about User 1?", "persona"),
    ( 9, "How articulate is User 1?",                    "persona"),
    (10, "What interests does User 1 show?",             "persona"),
    (11, "Does User 1 ask a lot of questions?",          "persona"),
    (12, "What character traits does User 1 display?",   "persona"),
    (13, "How does User 1 behave in conversations?",     "persona"),
    (14, "What is user1's vocabulary like?",             "persona"),
    (15, "Who is User 1 based on the conversations?",    "persona"),

    # ── Persona: User 2 (16-28) ───────────────────────────────────────────────
    (16, "What are the habits of User 2?",               "persona"),
    (17, "Describe User 2's personality",                "persona"),
    (18, "What is User 2's communication style?",        "persona"),
    (19, "What personal facts do we know about User 2?", "persona"),
    (20, "Does User 2 ask many questions?",              "persona"),
    (21, "How does User 2 express themselves?",          "persona"),
    (22, "What kind of person is user2?",                "persona"),
    (23, "Is User 2 expressive or reserved?",            "persona"),
    (24, "What topics does User 2 mention most?",        "persona"),
    (25, "Describe User 2's behavior in chats",          "persona"),
    (26, "What do we know about User 2's work life?",    "persona"),
    (27, "What locations does User 2 mention?",          "persona"),
    (28, "How does User 2's style differ from User 1?",  "persona"),

    # ── Topic (29-39) ─────────────────────────────────────────────────────────
    (29, "What topics were most discussed?",                   "topic"),
    (30, "What subjects came up in the conversations?",        "topic"),
    (31, "What did the users talk about?",                     "topic"),
    (32, "What are the main discussion subjects?",             "topic"),
    (33, "Which topics appear most frequently?",               "topic"),
    (34, "What themes were discussed?",                        "topic"),
    (35, "What topics about fitness came up?",                 "topic"),
    (36, "Were any work-related topics discussed?",            "topic"),
    (37, "What recurring themes exist in the conversations?",  "topic"),
    (38, "What subjects were most popular among the users?",   "topic"),
    (39, "What did they discuss about daily life?",            "topic"),

    # ── Summary (40-47) ───────────────────────────────────────────────────────
    (40, "Give me a summary of the conversations",              "summary"),
    (41, "What is an overview of the chats?",                   "summary"),
    (42, "Summarize the key points from the conversations",     "summary"),
    (43, "Give me a recap of what was discussed",               "summary"),
    (44, "What were the key highlights of the conversations?",  "summary"),
    (45, "What happened overall in these conversations?",       "summary"),
    (46, "Give me the gist of the chats",                       "summary"),
    (47, "What is the overall theme across all conversations?", "summary"),

    # ── All / general (48-50) ─────────────────────────────────────────────────
    (48, "Tell me everything about both users",              "all"),
    (49, "Give a complete analysis of the data",             "all"),
    (50, "What insights can be drawn from the knowledge base?", "all"),
]


# ── LLM availability check (independent of KB) ────────────────────────────────
def _check_llm() -> str:
    """Return error string if no LLM available, else empty string."""
    from src.llm import call_llm
    try:
        resp = call_llm("Reply with the single word: ok")
        if resp and len(resp.strip()) > 0:
            return ""
        return "call_llm returned empty response"
    except BaseException as e:
        return f"LLM call failed: {e}"


# ── KB check + optional auto-build ────────────────────────────────────────────
def _ensure_kb() -> str:
    """
    Return error string if KB can't be made ready, else empty string.
    Auto-builds from CSV/PDF if data exists but KB is empty.
    """
    from src.config import CSV_PATH, PDF_PATH

    # Step 1 — try to query KB state
    try:
        from src.kb_store import is_kb_populated
        populated = is_kb_populated()
    except BaseException as e:
        # ChromaDB may panic on a corrupt DB — wipe and retry once
        import shutil
        from src.config import CHROMA_DIR
        import src.kb_store as _kb
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _kb._chroma_client = None          # reset singleton
        try:
            populated = _kb.is_kb_populated()
        except BaseException as e2:
            return f"ChromaDB init failed even after wipe: {e2}"

    if populated:
        return ""

    # Step 2 — not populated; try auto-build if data exists
    has_data = os.path.exists(CSV_PATH) or os.path.exists(PDF_PATH)
    if not has_data:
        return "KB not populated and no CSV/PDF found in data/ — upload data first"

    print("\n[setup] KB is empty — auto-building from data files (this may take a minute)...")
    try:
        from src.pipeline import run_pipeline
        run_pipeline(force_rebuild=True)
        return ""
    except BaseException as e:
        return f"Pipeline auto-build failed: {e}"


# ── Session fixture ────────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def session_setup():
    # 1. LLM check
    llm_err = _check_llm()
    if llm_err:
        pytest.skip(f"No LLM available ({llm_err}) — start Ollama or set GROQ_API_KEY")

    # 2. KB check / auto-build
    kb_err = _ensure_kb()
    if kb_err:
        pytest.skip(kb_err)


@pytest.fixture(scope="session")
def answer_fn():
    from src.query_engine import answer_query
    return answer_query


# ── Helper ────────────────────────────────────────────────────────────────────
_LLM_ERROR_PHRASES = ("llm unavailable", "groq_api_key not set",
                      "please ensure", "api error")

def _looks_like_error(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _LLM_ERROR_PHRASES)


# ── 50 parametrized tests ─────────────────────────────────────────────────────
@pytest.mark.parametrize("qid,query,expected_route", QUERY_CASES,
                         ids=[f"Q{c[0]:02d}" for c in QUERY_CASES])
def test_query(qid, query, expected_route, answer_fn):
    result = answer_fn(query)

    # ── structure ──────────────────────────────────────────────────────────────
    assert isinstance(result, dict)
    for key in ("query", "route", "context", "answer"):
        assert key in result, f"Missing key '{key}' in result"

    # ── route ──────────────────────────────────────────────────────────────────
    assert result["route"] in ("persona", "topic", "summary", "all")
    assert result["route"] == expected_route, (
        f"Q{qid:02d}: expected route='{expected_route}' got='{result['route']}'\n"
        f"  query: {query}"
    )

    # ── context ────────────────────────────────────────────────────────────────
    assert result["context"].strip(), "Context is empty — KB may not be populated"
    assert result["context"] != "No relevant information found in knowledge base.", (
        f"Q{qid:02d}: KB returned no context for route='{expected_route}'"
    )

    # ── answer ─────────────────────────────────────────────────────────────────
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 10, (
        f"Q{qid:02d}: answer too short: {repr(result['answer'])}"
    )
    assert not _looks_like_error(result["answer"]), (
        f"Q{qid:02d}: answer looks like LLM error:\n  {result['answer'][:200]}"
    )

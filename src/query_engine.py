import json
import os
from typing import Dict, List
from src.config import KB_DIR
from src.kb_store import (retrieve_topics, retrieve_summaries,
                           retrieve_checkpoints, load_all_personas)
from src.llm import call_llm
from src.logger import logger


# ── Query classification ───────────────────────────────────────────────────────
PERSONA_KEYWORDS = {
    "persona", "person", "habit", "habits", "personality", "talk", "speak",
    "communicate", "communication", "style", "character", "who is", "what kind",
    "user 1", "user 2", "user1", "user2", "behave", "behaviour", "behavior"
}
TOPIC_KEYWORDS = {
    "topic", "topics", "discuss", "talk about",
    "what did", "when did", "subject", "theme", "themes", "recurring"
}
SUMMARY_KEYWORDS = {
    "summary", "summarize", "overview", "recap", "gist", "key points",
    "what happened", "overall", "highlights", "highlight"
}


def _classify_query(query: str) -> str:
    """Route query to correct KB layer."""
    q = query.lower()
    persona_score  = sum(1 for k in PERSONA_KEYWORDS  if k in q)
    topic_score    = sum(1 for k in TOPIC_KEYWORDS    if k in q)
    summary_score  = sum(1 for k in SUMMARY_KEYWORDS  if k in q)

    # Queries mentioning a specific user always belong to persona
    if any(u in q for u in ("user 1", "user 2", "user1", "user2")):
        persona_score += 1

    # Dict order defines tie-breaking: summary > persona > topic
    scores = {"summary": summary_score, "persona": persona_score, "topic": topic_score}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "all"
    return best


_MAX_CONTEXT_CHARS = 3500   # ~875 tokens — stays well under Groq free-tier TPM
_MAX_ITEM_CHARS    = 400    # max chars per individual KB chunk


def _clip(text: str, limit: int = _MAX_ITEM_CHARS) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _build_context(query: str, route: str) -> str:
    """Retrieve relevant KB chunks, clip each item, and cap total length."""
    parts = []

    if route in ("topic", "all"):
        topics = retrieve_topics(query, n_results=3)
        if topics:
            parts.append("=== RELEVANT TOPICS ===")
            for t in topics:
                parts.append(_clip(t["text"]))
                if t["metadata"].get("summary"):
                    parts.append(f"Summary: {_clip(t['metadata']['summary'], 200)}")

    if route in ("summary", "all"):
        summaries = retrieve_summaries(query, n_results=2)
        if summaries:
            parts.append("=== BATCH SUMMARIES ===")
            for s in summaries:
                parts.append(_clip(s["text"]))

        checkpoints = retrieve_checkpoints(query, n_results=1)
        if checkpoints:
            parts.append("=== CHRONOLOGICAL CHECKPOINTS ===")
            for c in checkpoints:
                parts.append(_clip(c["text"]))

    if route in ("persona", "all"):
        agg_path = os.path.join(KB_DIR, "aggregated_persona.json")
        if os.path.exists(agg_path):
            with open(agg_path, "r", encoding="utf-8") as f:
                agg = json.load(f)
            parts.append("=== USER PERSONAS ===")
            for user_key, data in agg.items():
                habits  = data.get("habits", [])[:8]
                facts   = data.get("personal_facts", [])[:10]
                persona = data.get("personality", "unknown")[:200]
                parts.append(f"\n{user_key}:")
                parts.append(f"  Habits: {', '.join(habits) or 'none noted'}")
                parts.append(f"  Facts: {', '.join(facts) or 'none noted'}")
                parts.append(f"  Personality: {persona}")
                parts.append(f"  Avg msg length: {data.get('avg_msg_length', 0)} words")

    context = "\n".join(parts) if parts else "No relevant information found in knowledge base."
    return context[:_MAX_CONTEXT_CHARS]


def answer_query(user_query: str) -> Dict:
    """Full query pipeline: route → retrieve → prompt → answer."""
    route   = _classify_query(user_query)
    context = _build_context(user_query, route)

    prompt = f"""You are an AI assistant that answers questions about chat conversations.
You have access to a knowledge base extracted from real conversations.

IMPORTANT RULES:
- Answer ONLY using the context provided below.
- Do NOT hallucinate or add information not in the context.
- If the context does not contain the answer, say "This information is not available in the knowledge base."
- Be concise and factual.

KNOWLEDGE BASE CONTEXT:
{context}

USER QUESTION: {user_query}

ANSWER:"""

    answer = call_llm(prompt)
    if not answer:
        answer = "LLM unavailable. Please ensure Groq API key is set or Ollama is running."

    return {
        "query":   user_query,
        "route":   route,
        "context": context,
        "answer":  answer
    }

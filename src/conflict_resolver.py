"""
conflict_resolver.py
─────────────────────
Drift-Aware Conflict Resolver for RAG — Part 2, Task 3

Problem:
  ChromaDB retrieves chunks by cosine similarity — blind to temporal
  contradictions. Same entity (e.g. "sister", "fitness") can appear
  across multiple drift segments with conflicting emotional context.
  Standard RAG sends all contradictory chunks to LLM → LLM picks one
  silently, averages vaguely, or hallucinates a narrative.

Solution:
  1. Extract entities from user query (spaCy NER + keyword fallback)
  2. Scan persona_drift.json for segments where entity appears
     — Method A: trigger.all_entities list (proper nouns, habits)
     — Method B: text scan of ChromaDB chunk text (common nouns: sister, mom)
  3. Detect contradictions using ALREADY COMPUTED drift signals:
     — Trigger A: polarity sign flip between consecutive matching segments
     — Trigger B: drift_magnitude > DRIFT_MAG_THRESHOLD (0.05)
  4. Targeted ChromaDB retrieval for ONLY those row ranges (not blind top-K)
  5. Build structured timeline context block for LLM
  6. LLM receives chronological timeline with contradiction flags

What this does NOT need:
  ✅ No extra model
  ✅ No recomputation of embeddings
  ✅ No antonym dictionary
  ✅ Uses persona_drift.json signals already computed by persona_drift.py
  ✅ Plugs between ChromaDB retrieval and LLM call in query_engine.py

Integration:
  from conflict_resolver import ConflictResolver
  resolver = ConflictResolver(drift_path, chroma_collections)
  context = resolver.resolve(query, user)   # replaces raw ChromaDB context
  if context is None:
      context = standard_chroma_retrieval(query)  # fallback
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DRIFT_JSON_PATH = BASE_DIR / "data" / "persona_drift.json"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Polarity sign flip = contradiction (pos → neg or neg → pos)
POLARITY_SIGN_FLIP = True

# Drift magnitude above this = significant behavioral shift at entity mention
DRIFT_MAG_THRESHOLD = 0.05

# Top-K chunks to pull from ChromaDB per matching segment row range
CHROMA_K_PER_SEGMENT = 2

# Max segments to include in LLM context (avoid token overload)
MAX_SEGMENTS_IN_CONTEXT = 6


# ─────────────────────────────────────────────
# ENTITY EXTRACTION FROM QUERY
# ─────────────────────────────────────────────

def extract_query_entities(query: str) -> list[str]:
    """
    Extract entities from user query using two methods:

    Method 1: spaCy NER (proper nouns, persons, places, orgs)
    Method 2: Keyword extraction (common nouns: sister, mom, boss, etc.)

    Returns deduplicated list of lowercase entity strings.
    """
    entities = set()

    # Method 1: spaCy NER (if available)
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None

        if nlp:
            doc = nlp(query)
            for ent in doc.ents:
                entities.add(ent.text.lower().strip())
            for token in doc:
                if token.pos_ in ("PROPN", "NOUN") and len(token.text) > 2:
                    entities.add(token.lemma_.lower().strip())
    except ImportError:
        pass

    # Method 2: Keyword extraction — common relational nouns + content words
    # These are unlikely to be caught by NER but are crucial for personal queries
    RELATIONAL_NOUNS = {
        "sister", "brother", "mom", "dad", "mother", "father", "friend",
        "boss", "colleague", "partner", "husband", "wife", "girlfriend",
        "boyfriend", "son", "daughter", "aunt", "uncle", "cousin",
        "manager", "teacher", "doctor", "therapist",
    }

    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    for word in query_words:
        if word in RELATIONAL_NOUNS:
            entities.add(word)

    # Also add any multi-word content after "about", "mention", "discuss"
    # e.g. "Did I mention anything about my fitness routine" → "fitness"
    about_match = re.search(r'\b(?:about|regarding|mention)\s+(?:my\s+)?(\w+)', query.lower())
    if about_match:
        entities.add(about_match.group(1))

    # Remove noise
    STOPWORDS = {"i", "my", "me", "the", "a", "an", "about", "did", "do",
                 "does", "any", "anything", "something", "mention", "said"}
    entities = {e for e in entities if e not in STOPWORDS and len(e) > 2}

    log.debug(f"[resolver] Query entities extracted: {entities}")
    return list(entities)


# ─────────────────────────────────────────────
# DRIFT SEGMENT MATCHING
# ─────────────────────────────────────────────

def find_matching_segments(
    entities: list[str],
    drift_data: dict,
    user: str,
) -> list[dict]:
    """
    Find drift segments where any query entity appears.

    Method A: Check trigger.all_entities (proper nouns, habits from NER dict)
    Method B: Check trigger.entity, trigger.personality, trigger.comm_style
              and all string fields in trigger for the entity term

    Returns list of matching segment dicts, sorted chronologically.
    """
    user_data = drift_data.get(user, {})
    timeline  = user_data.get("drift_timeline", [])

    if not timeline or not entities:
        return []

    matching = []

    for seg in timeline:
        trigger = seg.get("trigger") or {}
        matched_entity = None

        for entity in entities:
            entity_lower = entity.lower()

            # Method A: all_entities list
            all_ents = [e.lower() for e in (trigger.get("all_entities") or [])]
            if any(entity_lower in e or e in entity_lower for e in all_ents):
                matched_entity = entity
                break

            # Method A2: direct trigger.entity field
            t_ent = (trigger.get("entity") or "").lower()
            if entity_lower in t_ent or t_ent in entity_lower:
                matched_entity = entity
                break

            # Method B: broad text scan of all trigger string values
            trigger_text = " ".join(
                str(v).lower() for v in trigger.values()
                if isinstance(v, str)
            )
            if entity_lower in trigger_text:
                matched_entity = entity
                break

        if matched_entity:
            seg_copy = dict(seg)
            seg_copy["_matched_entity"] = matched_entity
            matching.append(seg_copy)

    # Sort by row_start (chronological)
    matching.sort(key=lambda s: s.get("row_start", 0))
    log.info(f"[resolver] {user}: {len(matching)} segments matched for entities {entities}")
    return matching


# ─────────────────────────────────────────────
# CONTRADICTION DETECTION (using existing drift signals)
# ─────────────────────────────────────────────

def detect_contradictions(segments: list[dict]) -> list[dict]:
    """
    Detect contradictions between consecutive matching segments.

    Two triggers:
      A: Polarity sign flip (positive → negative or negative → positive)
      B: drift_magnitude > DRIFT_MAG_THRESHOLD (significant behavioral shift)

    Returns list of contradiction dicts:
      {
        "between_segments": [seg_id_before, seg_id_after],
        "type": "polarity_flip" | "drift_shift" | "both",
        "polarity_before": float,
        "polarity_after":  float,
        "drift_magnitude": float,
        "description":     str (human-readable)
      }
    """
    contradictions = []

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        pol_before = prev.get("avg_polarity", 0.0)
        pol_after  = curr.get("avg_polarity", 0.0)
        drift_mag  = curr.get("drift_magnitude", 0.0)

        trigger_a = (pol_before * pol_after < 0)           # sign flip
        trigger_b = (drift_mag > DRIFT_MAG_THRESHOLD)

        if trigger_a or trigger_b:
            ctype = "both" if (trigger_a and trigger_b) else \
                    "polarity_flip" if trigger_a else "drift_shift"

            if trigger_a:
                direction = "positive → negative" if pol_before > 0 else "negative → positive"
                desc = (f"Sentiment flipped ({direction}) between "
                        f"segment {prev['segment']} (rows {prev['row_start']}–{prev['row_end']}) "
                        f"and segment {curr['segment']} (rows {curr['row_start']}–{curr['row_end']}). "
                        f"Polarity changed from {pol_before:+.3f} to {pol_after:+.3f}.")
            else:
                desc = (f"Significant behavioral shift (magnitude {drift_mag:.3f}) "
                        f"detected at segment {curr['segment']} "
                        f"(rows {curr['row_start']}–{curr['row_end']}).")

            contradictions.append({
                "between_segments": [prev["segment"], curr["segment"]],
                "type":             ctype,
                "polarity_before":  pol_before,
                "polarity_after":   pol_after,
                "drift_magnitude":  drift_mag,
                "description":      desc,
            })

    log.info(f"[resolver] {len(contradictions)} contradictions detected")
    return contradictions


# ─────────────────────────────────────────────
# TARGETED CHROMADB RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_chunks_for_segment(
    seg: dict,
    query: str,
    chroma_collections: dict,
    k: int = CHROMA_K_PER_SEGMENT,
) -> list[str]:
    """
    Pull ChromaDB chunks for a specific row range (row_start to row_end).

    Uses metadata filtering: where={"row_id": {"$gte": row_start, "$lte": row_end}}
    Falls back to semantic search with row range filter if metadata filter fails.

    chroma_collections: dict with keys "topics", "summaries", "checkpoints"
    Each value is a chromadb Collection object.
    """
    row_start = seg.get("row_start", 1)
    row_end   = seg.get("row_end",   row_start)
    chunks    = []

    for coll_name, collection in chroma_collections.items():
        if collection is None:
            continue
        try:
            # Try metadata range filter first
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where={
                    "$and": [
                        {"row_id": {"$gte": row_start}},
                        {"row_id": {"$lte": row_end}},
                    ]
                },
            )
            docs = results.get("documents", [[]])[0]
            chunks.extend(docs)

        except Exception:
            # Fallback: semantic search without range filter
            # (less precise but functional if metadata not set)
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=k,
                )
                docs = results.get("documents", [[]])[0]
                chunks.extend(docs)
            except Exception as e:
                log.warning(f"[resolver] ChromaDB query failed for {coll_name}: {e}")

    # Deduplicate
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)

    return unique_chunks[:k]


# ─────────────────────────────────────────────
# TIMELINE CONTEXT BUILDER (for LLM)
# ─────────────────────────────────────────────

def build_timeline_context(
    query:           str,
    entity:          str,
    segments:        list[dict],
    chunks_per_seg:  dict,      # {segment_id: [chunk_texts]}
    contradictions:  list[dict],
) -> str:
    """
    Build a structured timeline context block to send to the LLM.

    Structure:
      ENTITY: <entity>
      QUERY: <query>

      TIMELINE (oldest → newest):
        [Segment N | rows X–Y | tone | polarity]
        Retrieved text: "..."
        ⚠ CONTRADICTION: ...

      ANALYSIS INSTRUCTION:
        Answer based on the timeline above. Note any contradictions.
        Prioritize the most recent segment for current state.
        Acknowledge evolution explicitly.
    """
    if not segments:
        return ""

    lines = [
        f"━━━ CONFLICT-RESOLVED CONTEXT ━━━",
        f"Entity referenced: '{entity}'",
        f"User query: {query}",
        f"",
        f"TIMELINE (chronological, oldest → newest):",
        f"{'─' * 60}",
    ]

    # Build contradiction lookup by segment pair
    contra_map = {}
    for c in contradictions:
        pair = tuple(c["between_segments"])
        contra_map[pair] = c

    prev_seg_id = None

    for seg in segments[:MAX_SEGMENTS_IN_CONTEXT]:
        seg_id    = seg["segment"]
        row_start = seg["row_start"]
        row_end   = seg["row_end"]
        tone      = seg.get("tone", "unknown")
        polarity  = seg.get("avg_polarity", 0.0)
        trigger   = seg.get("trigger") or {}
        t_type    = trigger.get("type", "—")
        t_ent     = trigger.get("entity", "—") or "—"

        # Polarity label
        pol_label = "positive" if polarity > 0.1 else \
                    "negative" if polarity < -0.1 else "neutral"

        lines.append(
            f"\n[Segment {seg_id} | Rows {row_start}–{row_end} | "
            f"Tone: {tone} | Sentiment: {pol_label} ({polarity:+.3f})]"
        )

        if t_ent != "—":
            lines.append(f"  Trigger: {t_type} · Entity: {t_ent}")

        # Contradiction flag
        if prev_seg_id is not None:
            pair = (prev_seg_id, seg_id)
            if pair in contra_map:
                c = contra_map[pair]
                lines.append(f"  ⚠ CONTRADICTION DETECTED: {c['description']}")

        # Retrieved chunk text for this segment
        seg_chunks = chunks_per_seg.get(seg_id, [])
        if seg_chunks:
            for i, chunk in enumerate(seg_chunks[:2], 1):
                # Truncate long chunks
                chunk_preview = chunk[:400] + "..." if len(chunk) > 400 else chunk
                lines.append(f"  [{i}] {chunk_preview}")
        else:
            lines.append(f"  [No chunk retrieved for this segment]")

        prev_seg_id = seg_id

    lines.append(f"\n{'─' * 60}")

    # Summary of contradictions
    if contradictions:
        lines.append(f"\n⚠ TOTAL CONTRADICTIONS FOUND: {len(contradictions)}")
        for c in contradictions:
            lines.append(f"  • {c['description']}")
    else:
        lines.append(f"\n✓ No contradictions detected — consistent context across timeline.")

    # Most recent segment = most authoritative
    latest = segments[-1]
    lines.append(
        f"\n→ MOST RECENT MENTION: Segment {latest['segment']} "
        f"(rows {latest['row_start']}–{latest['row_end']}) | "
        f"Tone: {latest.get('tone', '?')} | "
        f"Polarity: {latest.get('avg_polarity', 0.0):+.3f}"
    )

    lines.append(
        f"\n━━━ INSTRUCTION TO LLM ━━━\n"
        f"Answer ONLY from the timeline above. Do not invent information.\n"
        f"If contradictions exist, acknowledge the evolution explicitly.\n"
        f"Prioritize the most recent segment for current state of '{entity}'.\n"
        f"Write in plain prose, not bullet points."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN RESOLVER CLASS
# ─────────────────────────────────────────────

class ConflictResolver:
    """
    Drift-Aware Conflict Resolver.

    Usage in query_engine.py:
        resolver = ConflictResolver(drift_path, chroma_collections)
        context = resolver.resolve(query, user="User_1")
        if context:
            # use conflict-resolved context
        else:
            # fall back to standard ChromaDB retrieval
    """

    def __init__(
        self,
        drift_path:         Path = DRIFT_JSON_PATH,
        chroma_collections: dict = None,
    ):
        self.drift_data         = self._load_drift(drift_path)
        self.chroma_collections = chroma_collections or {}

    def _load_drift(self, path: Path) -> dict:
        if not path.exists():
            log.warning(f"[resolver] persona_drift.json not found at {path}. "
                        f"Run persona_drift.py first.")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve(
        self,
        query:  str,
        user:   str = "User_1",
    ) -> Optional[str]:
        """
        Main entry point.

        Args:
            query: raw user query string
            user:  "User_1" or "User_2"

        Returns:
            Structured timeline context string if entity + drift match found.
            None if no match → caller should fall back to standard RAG retrieval.
        """
        if not self.drift_data:
            log.warning("[resolver] No drift data loaded. Falling back.")
            return None

        # Step 1: Extract entities from query
        entities = extract_query_entities(query)
        if not entities:
            log.info("[resolver] No entities extracted from query. Falling back.")
            return None

        # Step 2: Find matching drift segments
        matching_segments = find_matching_segments(entities, self.drift_data, user)
        if not matching_segments:
            log.info(f"[resolver] No drift segments matched for {entities}. Falling back.")
            return None

        if len(matching_segments) == 1:
            # Only one mention — no contradiction possible, still build context
            log.info("[resolver] Single segment match — no contradiction, building context.")

        # Step 3: Detect contradictions using drift signals
        contradictions = detect_contradictions(matching_segments)

        # Step 4: Targeted ChromaDB retrieval per matching segment
        chunks_per_seg = {}
        for seg in matching_segments[:MAX_SEGMENTS_IN_CONTEXT]:
            chunks = retrieve_chunks_for_segment(
                seg, query, self.chroma_collections
            )
            chunks_per_seg[seg["segment"]] = chunks

        # Also: Method B text scan — retrieve broadly and filter by entity term
        # This catches cases where entity appears in chunk text but not in drift trigger
        entity_str = entities[0] if entities else ""
        for seg in matching_segments[:MAX_SEGMENTS_IN_CONTEXT]:
            existing = chunks_per_seg.get(seg["segment"], [])
            if not existing:
                # Broad retrieval → text filter
                broad_chunks = self._text_scan_retrieval(query, entity_str)
                chunks_per_seg[seg["segment"]] = broad_chunks

        # Step 5: Build structured timeline context for LLM
        primary_entity = matching_segments[0].get("_matched_entity", entity_str)
        context = build_timeline_context(
            query          = query,
            entity         = primary_entity,
            segments       = matching_segments,
            chunks_per_seg = chunks_per_seg,
            contradictions = contradictions,
        )

        log.info(
            f"[resolver] Resolved: {len(matching_segments)} segments, "
            f"{len(contradictions)} contradictions, user={user}"
        )
        return context

    def _text_scan_retrieval(self, query: str, entity: str) -> list[str]:
        """
        Broad ChromaDB retrieval → filter chunks that contain the entity term.
        Fallback for common nouns not caught by NER (sister, mom, boss).
        """
        chunks = []
        for coll_name, collection in self.chroma_collections.items():
            if collection is None:
                continue
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=10,
                )
                docs = results.get("documents", [[]])[0]
                # Filter to chunks containing the entity term
                filtered = [d for d in docs if entity.lower() in d.lower()]
                chunks.extend(filtered[:2])
            except Exception as e:
                log.warning(f"[resolver] Text scan failed for {coll_name}: {e}")

        return chunks[:3]

    def has_drift_data(self) -> bool:
        return bool(self.drift_data)

    def get_stats(self, user: str = "User_1") -> dict:
        """Return summary stats for a user's drift data."""
        ud = self.drift_data.get(user, {})
        return {
            "total_segments":        len(ud.get("drift_timeline", [])),
            "total_drifts_detected": ud.get("total_drifts_detected", 0),
            "dominant_tone":         ud.get("dominant_tone", "unknown"),
        }


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

def _test():
    """
    Quick test without ChromaDB — verifies drift matching + contradiction detection.
    Loads persona_drift.json and runs resolver on sample queries.
    """
    logging.basicConfig(level=logging.INFO)

    drift_path = Path(__file__).resolve().parent.parent / "data" / "persona_drift.json"
    if not drift_path.exists():
        print(f"ERROR: persona_drift.json not found at {drift_path}")
        print("Run persona_drift.py first.")
        return

    resolver = ConflictResolver(drift_path=drift_path, chroma_collections={})

    TEST_QUERIES = [
        ("Did I mention anything about my fitness routine?", "User_1"),
        ("What did I say about Machu Picchu?",              "User_1"),
        ("Did I talk about my sister?",                     "User_1"),
        ("What were my career goals?",                      "User_2"),
        ("Tell me about the movies I mentioned",            "User_1"),
    ]

    print("\n═══ Conflict Resolver Test ═══\n")

    for query, user in TEST_QUERIES:
        print(f"Query: {query}")
        print(f"User:  {user}")
        print("─" * 60)

        context = resolver.resolve(query, user=user)

        if context:
            # Print first 800 chars of context
            print(context[:800])
            if len(context) > 800:
                print(f"... [truncated, total {len(context)} chars]")
        else:
            print("→ No drift match found. Standard RAG fallback would be used.")

        print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    _test()
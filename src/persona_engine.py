import re
from typing import Dict, List, Tuple
from src.logger import logger

# ── Lazy spaCy singleton ───────────────────────────────────────────────────────
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Lazy TextBlob singleton for sentiment analysis ──────────────────────────────
_textblob = None

def _get_textblob():
    global _textblob
    if _textblob is None:
        from textblob import TextBlob
        _textblob = TextBlob
    return _textblob


# ── NER label → readable fact prefix ──────────────────────────────────────────
_NER_LABELS = {
    "PERSON":      "mentions person",
    "ORG":         "mentions organization",
    "GPE":         "mentions location",
    "LOC":         "mentions location",
    "NORP":        "mentions group/nationality",
    "PRODUCT":     "mentions product",
    "MONEY":       "mentions money",
    "DATE":        "mentions date",
    "TIME":        "mentions time",
    "EVENT":       "mentions event",
    "WORK_OF_ART": "mentions work of art",
    "FAC":         "mentions facility",
    "LANGUAGE":    "mentions language",
}

# ── Keyword signal banks ───────────────────────────────────────────────────────
FOOD_WORDS    = {"food", "eat", "cook", "restaurant", "meal", "lunch", "dinner",
                 "breakfast", "recipe", "hungry", "delicious", "coffee", "drink"}
FITNESS_WORDS = {"gym", "workout", "exercise", "yoga", "run", "running", "hike",
                 "hiking", "sport", "sports", "fitness", "training", "walk"}
WORK_WORDS    = {"job", "work", "office", "meeting", "project", "boss", "career",
                 "business", "company", "client", "salary", "interview"}
FAMILY_WORDS  = {"family", "mom", "dad", "parent", "brother", "sister", "kids",
                 "children", "husband", "wife", "marriage", "relationship"}
SOCIAL_WORDS  = {"friend", "friends", "party", "social", "hang", "meet", "outing"}


def _count_signal(text_lower: str, word_set: set) -> int:
    return sum(1 for w in word_set if w in text_lower)


def _avg_msg_length(messages: List[str]) -> float:
    if not messages:
        return 0.0
    return round(sum(len(m.split()) for m in messages) / len(messages), 2)


def _question_ratio(messages: List[str]) -> float:
    if not messages:
        return 0.0
    return round(sum(1 for m in messages if "?" in m) / len(messages), 2)


def _emoji_count(messages: List[str]) -> int:
    pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF☀-⛿✀-➿]"
    )
    return sum(len(pattern.findall(m)) for m in messages)


def _vocabulary_richness(messages: List[str]) -> float:
    all_words = " ".join(messages).lower().split()
    if not all_words:
        return 0.0
    return round(len(set(all_words)) / len(all_words), 3)


def extract_statistical_persona(conv: Dict) -> Dict:
    """Pure stats-based persona per conversation."""
    result = {}
    for user_key, msg_list in [("User_1", conv["user1_msgs"]), ("User_2", conv["user2_msgs"])]:
        full_text = " ".join(msg_list).lower()
        result[user_key] = {
            "avg_msg_length": _avg_msg_length(msg_list),
            "question_ratio": _question_ratio(msg_list),
            "emoji_count":    _emoji_count(msg_list),
            "vocab_richness": _vocabulary_richness(msg_list),
            "msg_count":      len(msg_list),
            "signals": {
                "food_mentions":    _count_signal(full_text, FOOD_WORDS),
                "fitness_mentions": _count_signal(full_text, FITNESS_WORDS),
                "work_mentions":    _count_signal(full_text, WORK_WORDS),
                "family_mentions":  _count_signal(full_text, FAMILY_WORDS),
                "social_mentions":  _count_signal(full_text, SOCIAL_WORDS),
            }
        }
    return result


def _habits_from_signals(signals: Dict) -> List[str]:
    mapping = [
        ("food_mentions",    "discusses food/cooking"),
        ("fitness_mentions", "discusses fitness/exercise"),
        ("work_mentions",    "frequently mentions work"),
        ("family_mentions",  "discusses family"),
        ("social_mentions",  "socially engaged"),
    ]
    return [label for key, label in mapping if signals.get(key, 0) >= 2]


def _personality_from_stats(stats: Dict) -> str:
    traits = []
    if stats.get("question_ratio", 0) > 0.4:
        traits.append("inquisitive")
    if stats.get("emoji_count", 0) > 3:
        traits.append("expressive")
    vr = stats.get("vocab_richness", 0)
    if vr > 0.75:
        traits.append("articulate")
    al = stats.get("avg_msg_length", 0)
    if al > 20:
        traits.append("verbose")
    elif al < 5:
        traits.append("terse")
    return ", ".join(traits) if traits else "neutral"


def _communication_style_from_stats(stats: Dict) -> str:
    al = stats.get("avg_msg_length", 0)
    ec = stats.get("emoji_count", 0)
    length = "long" if al > 20 else ("short" if al < 5 else "medium-length")
    emoji  = "heavy emoji use" if ec > 5 else ("some emojis" if ec > 0 else "no emojis")
    style  = "casual" if ec > 2 else "neutral"
    return f"{style}, {length} messages, {emoji}"


# ── Semantic Frame Engine ──────────────────────────────────────────────────────
def extract_semantic_frames(messages: List[str]) -> Dict:
    """
    Extract relationship triples and role signals using dependency parsing.
    Returns dict with relationship triples and role signals.
    """
    nlp = _get_nlp()
    frames = {
        "relationship_triples": [],  # (subject, predicate, object)
        "role_signals": {},          # entity → roles/relationships
        "action_targets": {}         # action → target entities
    }
    
    seen_triples = set()
    
    for msg in messages:
        doc = nlp(msg)
        
        # Extract subject-verb-object triples from dependency parse
        for token in doc:
            if token.pos_ == "VERB":
                # Find subject
                subj = None
                for child in token.subtree:
                    if "nsubj" in child.dep_:
                        subj = child.text.lower()
                        break
                
                # Find direct object
                obj = None
                for child in token.subtree:
                    if "dobj" in child.dep_ or "attr" in child.dep_:
                        obj = child.text.lower()
                        break
                
                if subj and obj:
                    triple = (subj, token.text.lower(), obj)
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        frames["relationship_triples"].append(triple)
        
        # Extract entities and their relationships
        for ent in doc.ents:
            entity_name = ent.text.lower()
            ner_label = ent.label_
            
            # Role signals based on context
            if entity_name not in frames["role_signals"]:
                frames["role_signals"][entity_name] = {
                    "entity": entity_name,
                    "types": [],
                    "actions": [],
                    "modifiers": []
                }
            
            # Map NER to role types
            if ner_label == "PERSON":
                frames["role_signals"][entity_name]["types"].append("person_mentioned")
            elif ner_label == "ORG":
                frames["role_signals"][entity_name]["types"].append("organization_mentioned")
            elif ner_label == "GPE" or ner_label == "LOC":
                frames["role_signals"][entity_name]["types"].append("location_mentioned")
            
            # Track actions associated with this entity
            for token in doc:
                if token.pos_ == "VERB":
                    if ent.start <= token.i < ent.end or any(
                        child.i == token.i for child in ent.subtree
                    ):
                        if token.text.lower() not in frames["role_signals"][entity_name]["actions"]:
                            frames["role_signals"][entity_name]["actions"].append(token.text.lower())
    
    return frames


# ── Entity-scoped Sentiment Scoper ─────────────────────────────────────────────
def extract_entity_sentiments(messages: List[str]) -> Dict:
    """
    Extract sentiment tied to entities rather than whole sentences.
    Returns dict mapping entities to their associated sentiment polarity.
    """
    nlp = _get_nlp()
    TextBlob = _get_textblob()
    
    entity_sentiments = {}  # entity → {"polarity": float, "mentions": count, "contexts": []}
    
    for msg in messages:
        doc = nlp(msg)
        
        # Get overall sentence sentiment
        try:
            blob = TextBlob(msg)
            sentence_polarity = blob.sentiment.polarity  # -1 to 1
        except:
            sentence_polarity = 0.0
        
        # Scope sentiment to entities
        for ent in doc.ents:
            entity_name = ent.text.lower()
            
            # Check if there are sentiment indicators in the entity's sentence
            ent_start_char = ent.start_char
            ent_end_char = ent.end_char
            
            # Extract a window around the entity for local sentiment
            local_start = max(0, ent_start_char - 50)
            local_end = min(len(msg), ent_end_char + 50)
            local_context = msg[local_start:local_end]
            
            try:
                local_blob = TextBlob(local_context)
                entity_polarity = local_blob.sentiment.polarity
            except:
                entity_polarity = sentence_polarity
            
            if entity_name not in entity_sentiments:
                entity_sentiments[entity_name] = {
                    "polarity_sum": 0.0,
                    "mention_count": 0,
                    "contexts": [],
                    "sentiment_label": "neutral"
                }
            
            entity_sentiments[entity_name]["polarity_sum"] += entity_polarity
            entity_sentiments[entity_name]["mention_count"] += 1
            entity_sentiments[entity_name]["contexts"].append(local_context.strip())
    
    # Calculate average polarity and sentiment label
    for entity in entity_sentiments:
        data = entity_sentiments[entity]
        data["avg_polarity"] = round(data["polarity_sum"] / max(data["mention_count"], 1), 3)
        
        avg_pol = data["avg_polarity"]
        if avg_pol > 0.1:
            data["sentiment_label"] = "positive"
        elif avg_pol < -0.1:
            data["sentiment_label"] = "negative"
        else:
            data["sentiment_label"] = "neutral"
    
    return entity_sentiments


def extract_ner_persona(conv: Dict) -> Dict:
    """spaCy NER + stats-derived persona — no LLM required."""
    nlp    = _get_nlp()
    stats  = extract_statistical_persona(conv)
    result = {}

    for user_key, msg_list in [("User_1", conv["user1_msgs"]), ("User_2", conv["user2_msgs"])]:
        if not msg_list:
            result[user_key] = {
                "habits": [], "personal_facts": [],
                "personality": "unknown", "communication_style": "unknown"
            }
            continue

        personal_facts = []
        seen = set()
        for msg in msg_list:
            doc = nlp(msg)
            for ent in doc.ents:
                prefix = _NER_LABELS.get(ent.label_)
                if prefix:
                    fact = f"{prefix}: {ent.text}"
                    if fact not in seen:
                        seen.add(fact)
                        personal_facts.append(fact)

        user_stats = stats[user_key]
        result[user_key] = {
            "habits":             _habits_from_signals(user_stats["signals"]),
            "personal_facts":     personal_facts,
            "personality":        _personality_from_stats(user_stats),
            "communication_style": _communication_style_from_stats(user_stats),
        }
    return result


def extract_semantic_frame_persona(messages: List[str]) -> Dict:
    """
    Extract semantic frames, relationship triples, and role signals.
    """
    frames = extract_semantic_frames(messages)
    
    return {
        "relationship_triples": frames["relationship_triples"][:10],  # Top 10 triples
        "role_signals": frames["role_signals"],
        "action_targets": frames["action_targets"]
    }


def extract_entity_sentiment_persona(messages: List[str]) -> Dict:
    """
    Extract entity-scoped sentiments.
    """
    entity_sentiments = extract_entity_sentiments(messages)
    
    # Group by sentiment
    positive_entities = [
        e for e, data in entity_sentiments.items()
        if data["sentiment_label"] == "positive"
    ]
    negative_entities = [
        e for e, data in entity_sentiments.items()
        if data["sentiment_label"] == "negative"
    ]
    neutral_entities = [
        e for e, data in entity_sentiments.items()
        if data["sentiment_label"] == "neutral"
    ]
    
    return {
        "entity_sentiments": entity_sentiments,
        "positive_entities": positive_entities[:10],
        "negative_entities": negative_entities[:10],
        "neutral_entities": neutral_entities[:10]
    }


def extract_full_persona(conv: Dict) -> Dict:
    """
    Full persona extraction pipeline:
    1. Statistical engine
    2. NER engine
    3. Semantic frame engine
    4. Entity-scoped sentiment scoper
    """
    stats = extract_statistical_persona(conv)
    ner   = extract_ner_persona(conv)

    combined = {}
    for user_key in ["User_1", "User_2"]:
        msg_list = conv.get(f"user{'1' if user_key == 'User_1' else '2'}_msgs", [])
        
        # Extract semantic frames and entity sentiments for this user
        semantic_frames = extract_semantic_frame_persona(msg_list) if msg_list else {}
        entity_sentiments = extract_entity_sentiment_persona(msg_list) if msg_list else {}
        
        combined[user_key] = {
            "stats":      stats.get(user_key, {}),
            "ner":        ner.get(user_key, {}),
            "semantic_frames": semantic_frames,
            "entity_sentiments": entity_sentiments,
            "conv_id":    conv["conv_id"]
        }
    return combined


def aggregate_persona_across_batches(all_persona_entries: List[Dict]) -> Dict:
    """Aggregate per-batch persona entries into a final user profile."""
    aggregated = {
        "User_1": {
            "habits": set(), "personal_facts": set(),
            "personality_notes": [], "total_msgs": 0,
            "avg_msg_length_sum": 0.0, "conv_count": 0,
            "relationship_triples": [], "positive_entities": set(),
            "negative_entities": set(), "role_signals": {}
        },
        "User_2": {
            "habits": set(), "personal_facts": set(),
            "personality_notes": [], "total_msgs": 0,
            "avg_msg_length_sum": 0.0, "conv_count": 0,
            "relationship_triples": [], "positive_entities": set(),
            "negative_entities": set(), "role_signals": {}
        },
    }

    for entry in all_persona_entries:
        for user_key in ["User_1", "User_2"]:
            data  = entry.get(user_key, {})
            # Accept both new "ner" key and legacy "llm" key
            ner   = data.get("ner", data.get("llm", {}))
            stats = data.get("stats", {})
            semantic_frames = data.get("semantic_frames", {})
            entity_sentiments = data.get("entity_sentiments", {})

            for h in ner.get("habits", []):
                aggregated[user_key]["habits"].add(h)
            for f in ner.get("personal_facts", []):
                aggregated[user_key]["personal_facts"].add(f)
            p = ner.get("personality", "")
            if p and p not in ("unknown", "neutral"):
                aggregated[user_key]["personality_notes"].append(p)

            aggregated[user_key]["total_msgs"]         += stats.get("msg_count", 0)
            aggregated[user_key]["avg_msg_length_sum"] += stats.get("avg_msg_length", 0)
            aggregated[user_key]["conv_count"]         += 1
            
            # Aggregate semantic frames
            for triple in semantic_frames.get("relationship_triples", []):
                aggregated[user_key]["relationship_triples"].append(triple)
            
            # Aggregate role signals
            for entity, roles in semantic_frames.get("role_signals", {}).items():
                if entity not in aggregated[user_key]["role_signals"]:
                    aggregated[user_key]["role_signals"][entity] = {
                        "types": set(), "actions": set(), "occurrences": 0
                    }
                aggregated[user_key]["role_signals"][entity]["types"].update(roles.get("types", []))
                aggregated[user_key]["role_signals"][entity]["actions"].update(roles.get("actions", []))
                aggregated[user_key]["role_signals"][entity]["occurrences"] += 1
            
            # Aggregate entity sentiments
            for entity in entity_sentiments.get("positive_entities", []):
                aggregated[user_key]["positive_entities"].add(entity)
            for entity in entity_sentiments.get("negative_entities", []):
                aggregated[user_key]["negative_entities"].add(entity)

    final = {}
    for user_key in ["User_1", "User_2"]:
        agg   = aggregated[user_key]
        count = max(agg["conv_count"], 1)
        
        # Convert role signals sets back to lists
        role_signals_final = {}
        for entity, roles in agg["role_signals"].items():
            role_signals_final[entity] = {
                "types": list(roles["types"]),
                "actions": list(roles["actions"]),
                "occurrences": roles["occurrences"]
            }
        
        final[user_key] = {
            "habits":         list(agg["habits"]),
            "personal_facts": list(agg["personal_facts"]),
            "personality":    " | ".join(agg["personality_notes"][:5]),
            "avg_msg_length": round(agg["avg_msg_length_sum"] / count, 2),
            "total_msgs":     agg["total_msgs"],
            "relationship_triples": agg["relationship_triples"][:20],
            "positive_entities": list(agg["positive_entities"])[:20],
            "negative_entities": list(agg["negative_entities"])[:20],
            "role_signals": role_signals_final
        }
    return final

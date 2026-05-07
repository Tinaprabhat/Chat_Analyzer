import re
from typing import Dict, List
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


def extract_full_persona(conv: Dict) -> Dict:
    """Stats engine + spaCy NER combined."""
    stats = extract_statistical_persona(conv)
    ner   = extract_ner_persona(conv)

    combined = {}
    for user_key in ["User_1", "User_2"]:
        combined[user_key] = {
            "stats":   stats.get(user_key, {}),
            "ner":     ner.get(user_key, {}),
            "conv_id": conv["conv_id"]
        }
    return combined


def aggregate_persona_across_batches(all_persona_entries: List[Dict]) -> Dict:
    """Aggregate per-batch persona entries into a final user profile."""
    aggregated = {
        "User_1": {"habits": set(), "personal_facts": set(),
                   "personality_notes": [], "total_msgs": 0,
                   "avg_msg_length_sum": 0.0, "conv_count": 0},
        "User_2": {"habits": set(), "personal_facts": set(),
                   "personality_notes": [], "total_msgs": 0,
                   "avg_msg_length_sum": 0.0, "conv_count": 0},
    }

    for entry in all_persona_entries:
        for user_key in ["User_1", "User_2"]:
            data  = entry.get(user_key, {})
            # Accept both new "ner" key and legacy "llm" key
            ner   = data.get("ner", data.get("llm", {}))
            stats = data.get("stats", {})

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

    final = {}
    for user_key in ["User_1", "User_2"]:
        agg   = aggregated[user_key]
        count = max(agg["conv_count"], 1)
        final[user_key] = {
            "habits":         list(agg["habits"]),
            "personal_facts": list(agg["personal_facts"]),
            "personality":    " | ".join(agg["personality_notes"][:5]),
            "avg_msg_length": round(agg["avg_msg_length_sum"] / count, 2),
            "total_msgs":     agg["total_msgs"]
        }
    return final

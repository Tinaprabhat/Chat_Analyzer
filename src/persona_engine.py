# persona_engine.py — Multilayer Persona Engine V2


import re
import math
from collections import Counter, defaultdict
from datetime import datetime

import spacy
from textblob import TextBlob

# =========================================================
# LOAD NLP MODEL
# =========================================================

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# =========================================================
# SIGNAL BANKS
# =========================================================

FOOD_WORDS = {
    "cook", "cooking", "food", "eat", "eating", "chef", "meal",
    "restaurant", "kitchen", "recipe", "bake", "baking", "dinner",
    "lunch", "breakfast", "dish", "cuisine", "drink", "coffee"
}

PET_WORDS = {
    "dog", "cat", "pet", "animal", "fish", "bird", "rabbit",
    "puppy", "kitten", "hamster", "turtle"
}

FITNESS_WORDS = {
    "gym", "workout", "exercise", "fitness", "run", "running",
    "yoga", "sport", "training", "weights", "hike", "hiking", "cycling"
}

TECH_WORDS = {
    "ai", "ml", "llm", "rag", "agent", "python", "model",
    "transformer", "neural", "system", "architecture", "backend",
    "frontend", "deployment", "docker", "fastapi", "streamlit"
}

CAREER_WORDS = {
    "job", "career", "internship", "salary", "placement",
    "resume", "hire", "company", "interview"
}

LEARNING_WORDS = {
    "learn", "study", "understand", "deep dive", "explain",
    "teach", "notes", "roadmap", "syllabus"
}

EMOTION_POSITIVE = {
    "love", "enjoy", "great", "awesome", "amazing",
    "happy", "excited", "fun"
}

EMOTION_NEGATIVE = {
    "hate", "frustrated", "confused", "bad",
    "stress", "difficult", "hard", "annoying"
}

PREFERENCE_PATTERNS = {
    "concise_answers": [
        r"short answer",
        r"concise",
        r"brief",
        r"keep it short"
    ],
    "detailed_answers": [
        r"deep dive",
        r"detailed",
        r"full explanation",
        r"step by step"
    ],
    "practical_focus": [
        r"industry relevant",
        r"real world",
        r"production",
        r"deploy"
    ],
    "systems_thinking": [
        r"architecture",
        r"workflow",
        r"pipeline",
        r"system design"
    ]
}

# =========================================================
# HELPERS
# =========================================================


def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())



def safe_div(a, b):
    return a / b if b else 0



def count_matches(tokens, vocab):
    return sum(1 for t in tokens if t in vocab)


# =========================================================
# 1. STATS ENGINE
# =========================================================


class StatsEngine:

    def extract(self, messages):
        joined = " ".join(messages)
        tokens = tokenize(joined)

        total_words = len(tokens)
        total_msgs = max(len(messages), 1)

        avg_msg_length = safe_div(total_words, total_msgs)

        question_ratio = safe_div(
            sum(1 for m in messages if "?" in m),
            total_msgs
        )

        emoji_count = len(re.findall(
            r"[😀-🙏🌀-🗿🚀-🛿🇠-🇿]",
            joined
        ))

        vocab_richness = safe_div(len(set(tokens)), total_words)

        return {
            "message_count": total_msgs,
            "total_words": total_words,
            "avg_msg_length": round(avg_msg_length, 2),
            "question_ratio": round(question_ratio, 2),
            "emoji_count": emoji_count,
            "vocab_richness": round(vocab_richness, 2)
        }


# =========================================================
# 2. EVENT EXTRACTION LAYER
# =========================================================


class EventExtractionLayer:

    def extract(self, messages):
        events = []

        for msg in messages:
            text = clean_text(msg)
            tokens = tokenize(text)

            if count_matches(tokens, TECH_WORDS) >= 2:
                events.append({
                    "type": "interest",
                    "value": "technology_ai"
                })

            if count_matches(tokens, CAREER_WORDS) >= 1:
                events.append({
                    "type": "goal",
                    "value": "career_growth"
                })

            if count_matches(tokens, LEARNING_WORDS) >= 1:
                events.append({
                    "type": "behavior",
                    "value": "active_learning"
                })

            if any(w in tokens for w in EMOTION_POSITIVE):
                events.append({
                    "type": "emotion",
                    "value": "positive"
                })

            if any(w in tokens for w in EMOTION_NEGATIVE):
                events.append({
                    "type": "emotion",
                    "value": "negative"
                })

            # Goal detection
            if re.search(r"want to|aim to|trying to", text.lower()):
                events.append({
                    "type": "goal_expression",
                    "value": text[:120]
                })

        return events


# =========================================================
# 3. PREFERENCE MEMORY LAYER
# =========================================================


class PreferenceMemoryLayer:

    def extract(self, messages):
        prefs = defaultdict(int)

        for msg in messages:
            text = clean_text(msg).lower()

            for pref_name, patterns in PREFERENCE_PATTERNS.items():
                for p in patterns:
                    if re.search(p, text):
                        prefs[pref_name] += 1

        final_prefs = []

        for k, v in prefs.items():
            confidence = min(1.0, 0.4 + (v * 0.1))

            final_prefs.append({
                "preference": k,
                "strength": round(confidence, 2),
                "evidence_count": v
            })

        return final_prefs


# =========================================================
# 4. SENTIMENT ENGINE
# =========================================================


class SentimentEngine:

    def extract(self, messages):

        sentiments = []

        for msg in messages:
            text = clean_text(msg)

            polarity = TextBlob(text).sentiment.polarity

            if polarity > 0.2:
                label = "positive"
            elif polarity < -0.2:
                label = "negative"
            else:
                label = "neutral"

            sentiments.append({
                "text": text[:120],
                "polarity": round(polarity, 2),
                "label": label
            })

        avg_sentiment = safe_div(
            sum(s["polarity"] for s in sentiments),
            len(sentiments)
        )

        if avg_sentiment > 0.2:
            overall = "optimistic"
        elif avg_sentiment < -0.2:
            overall = "frustrated"
        else:
            overall = "balanced"

        return {
            "overall_sentiment": overall,
            "average_polarity": round(avg_sentiment, 2),
            "samples": sentiments[:10]
        }


# =========================================================
# SEMANTIC FRAME EXTRACTION
# =========================================================


class SemanticFrameExtractor:

    def extract(self, messages):

        triples = []

        if not nlp:
            return triples

        for msg in messages:
            doc = nlp(msg)

            for token in doc:

                if token.dep_ == "ROOT":
                    subj = None
                    obj = None

                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = child.text
                        if child.dep_ in ["dobj", "pobj", "attr"] and child.pos_ in ("NOUN", "PROPN"):
                            obj = child.text

                    if subj and obj:
                        triples.append({
                            "subject": subj.lower(),
                            "relation": token.text.lower(),
                            "object": obj.lower()
                        })

        return triples[:100]


# =========================================================
# 5. TRAIT INFERENCE LAYER
# =========================================================


class TraitInferenceLayer:

    def infer(self,
              stats,
              events,
              preferences,
              sentiment,
              semantic_frames):

        traits = []

        event_counter = Counter(
            [e["value"] for e in events]
        )

        # =================================================
        # Analytical thinker
        # =================================================

        if (
            stats["question_ratio"] > 0.3 and
            stats["vocab_richness"] > 0.55
        ):
            traits.append({
                "trait": "analytical_thinker",
                "confidence": 0.82,
                "evidence": [
                    "high question ratio",
                    "rich vocabulary"
                ]
            })

        # =================================================
        # Systems oriented
        # =================================================

        systems_pref = any(
            p["preference"] == "systems_thinking"
            for p in preferences
        )

        if systems_pref or event_counter["technology_ai"] >= 3:
            traits.append({
                "trait": "systems_oriented",
                "confidence": 0.84,
                "evidence": [
                    "architecture discussions",
                    "technology focus"
                ]
            })

        # =================================================
        # Career ambitious
        # =================================================

        if event_counter["career_growth"] >= 2:
            traits.append({
                "trait": "career_ambitious",
                "confidence": 0.78,
                "evidence": [
                    "repeated career discussions"
                ]
            })

        # =================================================
        # Active learner
        # =================================================

        if event_counter["active_learning"] >= 2:
            traits.append({
                "trait": "active_learner",
                "confidence": 0.88,
                "evidence": [
                    "frequent learning-oriented queries"
                ]
            })

        # =================================================
        # Communication style traits
        # =================================================

        if stats["avg_msg_length"] < 12:
            traits.append({
                "trait": "concise_communicator",
                "confidence": 0.72,
                "evidence": ["short messages"]
            })

        if stats["avg_msg_length"] > 35:
            traits.append({
                "trait": "detail_oriented",
                "confidence": 0.74,
                "evidence": ["long detailed messages"]
            })

        # =================================================
        # Emotional profile
        # =================================================

        if sentiment["overall_sentiment"] == "optimistic":
            traits.append({
                "trait": "positive_outlook",
                "confidence": 0.69,
                "evidence": ["positive sentiment trend"]
            })

        if sentiment["overall_sentiment"] == "frustrated":
            traits.append({
                "trait": "problem_focused",
                "confidence": 0.64,
                "evidence": ["frequent frustration indicators"]
            })

        # =================================================
        # Semantic frame reasoning
        # =================================================

        ai_relations = sum(
            1 for t in semantic_frames
            if t["object"] in ["ai", "ml", "architecture", "system"]
        )

        if ai_relations >= 2:
            traits.append({
                "trait": "deep_technical_interest",
                "confidence": 0.85,
                "evidence": [
                    "repeated semantic relations to AI systems"
                ]
            })

        return traits


# =========================================================
# PERSONA ENGINE V2
# =========================================================


class PersonaEngineV2:

    def __init__(self):

        self.stats_engine = StatsEngine()
        self.event_layer = EventExtractionLayer()
        self.preference_layer = PreferenceMemoryLayer()
        self.sentiment_engine = SentimentEngine()
        self.semantic_engine = SemanticFrameExtractor()
        self.trait_engine = TraitInferenceLayer()

    def build_persona(self, messages, user_id=None):

        if not messages:
            return {}

        # =================================================
        # 1. STATS
        # =================================================

        stats = self.stats_engine.extract(messages)

        # =================================================
        # 2. EVENTS
        # =================================================

        events = self.event_layer.extract(messages)

        # =================================================
        # 3. PREFERENCES
        # =================================================

        preferences = self.preference_layer.extract(messages)

        # =================================================
        # 4. SENTIMENT
        # =================================================

        sentiment = self.sentiment_engine.extract(messages)

        # =================================================
        # 5. SEMANTIC FRAMES
        # =================================================

        semantic_frames = self.semantic_engine.extract(messages)

        # =================================================
        # 6. TRAIT INFERENCE
        # =================================================

        traits = self.trait_engine.infer(
            stats,
            events,
            preferences,
            sentiment,
            semantic_frames
        )

        # =================================================
        # PERSONA OBJECT
        # =================================================

        persona = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),

            "stats_profile": stats,

            "event_memory": events,

            "preferences": preferences,

            "sentiment_profile": sentiment,

            "semantic_memory": {
                "relationship_triples": semantic_frames
            },

            "traits": traits,

            "identity_summary": self._build_identity_summary(
                traits,
                preferences,
                sentiment
            )
        }

        return persona

    # =====================================================
    # IDENTITY SUMMARY
    # =====================================================

    def _build_identity_summary(self,
                                traits,
                                preferences,
                                sentiment):

        trait_names = [t["trait"] for t in traits]

        pref_names = [
            p["preference"]
            for p in preferences
        ]

        summary = {
            "primary_traits": trait_names[:5],
            "communication_preferences": pref_names,
            "overall_emotional_pattern": sentiment[
                "overall_sentiment"
            ]
        }

        return summary


# =========================================================
# ENGINE SINGLETONS  (shared across all module-level calls)
# =========================================================

_stats_engine     = StatsEngine()
_event_layer      = EventExtractionLayer()
_preference_layer = PreferenceMemoryLayer()
_sentiment_engine = SentimentEngine()
_semantic_engine  = SemanticFrameExtractor()
_trait_engine     = TraitInferenceLayer()


# =========================================================
# MODULE-LEVEL API (used by pipeline and tests)
# =========================================================


def extract_statistical_persona(conv):
    """Statistics-based persona for User_1 and User_2 via StatsEngine."""
    result = {}
    for user_key, msgs in [("User_1", conv.get("user1_msgs", [])),
                            ("User_2", conv.get("user2_msgs", []))]:
        if not msgs:
            result[user_key] = {
                "avg_msg_length": 0.0, "question_ratio": 0.0,
                "emoji_count": 0, "vocab_richness": 0.0, "msg_count": 0,
                "signals": {"food_mentions": 0, "fitness_mentions": 0, "pet_mentions": 0},
            }
            continue

        stats  = _stats_engine.extract(msgs)
        tokens = tokenize(" ".join(msgs))

        result[user_key] = {
            "avg_msg_length": stats["avg_msg_length"],
            "question_ratio": stats["question_ratio"],
            "emoji_count":    stats["emoji_count"],
            "vocab_richness": stats["vocab_richness"],
            "msg_count":      stats["message_count"],
            "signals": {
                "food_mentions":    count_matches(tokens, FOOD_WORDS),
                "fitness_mentions": count_matches(tokens, FITNESS_WORDS),
                "pet_mentions":     count_matches(tokens, PET_WORDS),
            },
        }
    return result


def extract_ner_persona(conv):
    """NER + heuristic persona wired through EventExtractionLayer,
    SentimentEngine, and PreferenceMemoryLayer."""
    user1_msgs = conv.get("user1_msgs", [])
    user2_msgs = conv.get("user2_msgs", [])

    def avg_len(msgs):
        if not msgs:
            return 0.0
        return sum(len(m.split()) for m in msgs) / len(msgs)

    u1_avg = avg_len(user1_msgs)
    u2_avg = avg_len(user2_msgs)
    overall_avg = (u1_avg + u2_avg) / 2 or 1.0

    result = {}
    for user_key, msgs in [("User_1", user1_msgs), ("User_2", user2_msgs)]:
        if not msgs:
            result[user_key] = {
                "habits": [], "personal_facts": [],
                "personality": "neutral", "communication_style": "unknown",
                "preferences": [],
            }
            continue

        joined = " ".join(msgs)
        tokens = tokenize(joined)

        # --- Habits: domain signals + EventExtractionLayer for tech/career/learning ---
        events      = _event_layer.extract(msgs)
        event_types = {e["value"] for e in events}

        habits = []
        if count_matches(tokens, FOOD_WORDS) >= 1:
            habits.append("discusses food/cooking")
        if count_matches(tokens, FITNESS_WORDS) >= 1:
            habits.append("discusses fitness/exercise")
        if count_matches(tokens, PET_WORDS) >= 1:
            habits.append("discusses pets")
        if "technology_ai" in event_types:
            habits.append("discusses technology")
        if "career_growth" in event_types:
            habits.append("discusses career")
        if "active_learning" in event_types:
            habits.append("discusses learning")

        # --- Personal facts via spaCy NER ---
        personal_facts = []
        if nlp:
            doc = nlp(joined)
            label_map = {
                "PERSON": "person", "GPE": "location", "LOC": "location",
                "ORG": "organization", "NORP": "group", "FAC": "place",
            }
            for ent in doc.ents:
                if ent.label_ in label_map:
                    fact = f"mentions {label_map[ent.label_]}: {ent.text}"
                    if fact not in personal_facts:
                        personal_facts.append(fact)

        # --- Personality from SentimentEngine ---
        sentiment = _sentiment_engine.extract(msgs)
        if sentiment["overall_sentiment"] == "optimistic":
            personality = "expressive"
        elif sentiment["overall_sentiment"] == "frustrated":
            personality = "cautious"
        else:
            personality = "inquisitive"

        # --- Preferences from PreferenceMemoryLayer ---
        prefs = _preference_layer.extract(msgs)

        # --- Communication style (relative length thresholds) ---
        user_avg  = avg_len(msgs)
        emoji_tag = "some emojis" if re.search(r"[😀-🙏🌀-🗿🚀-🛿🇠-🇿]", joined) else "no emojis"

        if user_avg < overall_avg * 0.6:
            length_tag = "terse"
        elif user_avg > overall_avg * 1.4:
            length_tag = "verbose"
        else:
            length_tag = "medium-length messages"

        result[user_key] = {
            "habits":             habits,
            "personal_facts":     personal_facts,
            "personality":        personality,
            "communication_style": f"casual, {length_tag}, {emoji_tag}",
            "preferences":        prefs,
        }
    return result


def extract_semantic_frames(messages):
    """Triples via SemanticFrameExtractor (NOUN/PROPN objects only).
    Returns {"relationship_triples": [(subj, rel, obj), ...]}"""
    raw = _semantic_engine.extract(messages)
    triples = [(t["subject"], t["relation"], t["object"]) for t in raw]
    return {"relationship_triples": triples}


def extract_full_persona(conv):
    """Full pipeline persona: stats + NER + events + preferences + traits
    via all engine layers.  'ner' key preserved for aggregate compatibility."""
    user1_msgs = conv.get("user1_msgs", [])
    user2_msgs = conv.get("user2_msgs", [])
    all_msgs   = user1_msgs + user2_msgs

    stats_map = extract_statistical_persona(conv)
    ner_map   = extract_ner_persona(conv)
    semantic_frames = _semantic_engine.extract(all_msgs)

    result = {}
    for user_key, msgs in [("User_1", user1_msgs), ("User_2", user2_msgs)]:
        stats     = _stats_engine.extract(msgs) if msgs else _stats_engine.extract([""])
        events    = _event_layer.extract(msgs) if msgs else []
        prefs     = _preference_layer.extract(msgs) if msgs else []
        sentiment = _sentiment_engine.extract(msgs) if msgs else _sentiment_engine.extract([""])
        traits    = _trait_engine.infer(stats, events, prefs, sentiment, semantic_frames)

        result[user_key] = {
            "stats":       stats_map.get(user_key, {}),
            "ner":         ner_map.get(user_key, {}),
            "events":      events,
            "preferences": prefs,
            "traits":      traits,
        }
    return result


def aggregate_persona_across_batches(personas):
    """Merge {user_key: {ner/llm, stats, ...}} dicts from multiple batches."""
    merged = {}
    for persona_dict in personas:
        for user_key, data in persona_dict.items():
            if user_key not in merged:
                merged[user_key] = {
                    "habits": [], "personal_facts": [],
                    "personality": [], "communication_style": [],
                }
            ner_data = data.get("ner") or data.get("llm") or {}
            for habit in ner_data.get("habits", []):
                if habit not in merged[user_key]["habits"]:
                    merged[user_key]["habits"].append(habit)
            for fact in ner_data.get("personal_facts", []):
                if fact not in merged[user_key]["personal_facts"]:
                    merged[user_key]["personal_facts"].append(fact)
            if ner_data.get("personality"):
                merged[user_key]["personality"].append(ner_data["personality"])
            if ner_data.get("communication_style"):
                merged[user_key]["communication_style"].append(ner_data["communication_style"])

    for user_key in merged:
        personalities = merged[user_key]["personality"]
        merged[user_key]["personality"] = (
            Counter(personalities).most_common(1)[0][0] if personalities else "neutral"
        )
        styles = merged[user_key]["communication_style"]
        merged[user_key]["communication_style"] = (
            Counter(styles).most_common(1)[0][0] if styles else "casual"
        )
    return merged


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    msgs = [
        "I want to become an AI systems architect.",
        "Give detailed explanation of RAG pipelines.",
        "I love building ML systems and deployment workflows.",
        "Can you explain observability in production AI systems?",
        "I am frustrated with DSA but enjoy AI architecture."
    ]

    engine = PersonaEngineV2()

    persona = engine.build_persona(
        msgs,
        user_id="demo_user"
    )

    from pprint import pprint
    pprint(persona)


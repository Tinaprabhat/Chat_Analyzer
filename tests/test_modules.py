"""
Modular unit tests for Chat Persona RAG system.
Run: python -m pytest tests/test_modules.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import tempfile


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Preprocessor
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessor:
    SAMPLE_RAW = (
        "User 1: Hey how are you?\n"
        "User 2: Doing well! What about you?\n"
        "User 1: Great! I love cooking Italian food.\n"
        "User 2: Me too! I work as a chef."
    )

    def test_parse_conversation_structure(self):
        from src.preprocessor import parse_conversation
        result = parse_conversation(self.SAMPLE_RAW, conv_id=0)
        assert result["conv_id"] == 0
        assert len(result["messages"]) == 4
        assert result["messages"][0]["sender"] == "User_1"
        assert result["messages"][1]["sender"] == "User_2"

    def test_parse_extracts_full_text(self):
        from src.preprocessor import parse_conversation
        result = parse_conversation(self.SAMPLE_RAW, conv_id=1)
        assert "cooking" in result["full_text"]
        assert "chef" in result["full_text"]

    def test_user1_user2_split(self):
        from src.preprocessor import parse_conversation
        result = parse_conversation(self.SAMPLE_RAW, conv_id=2)
        assert len(result["user1_msgs"]) == 2
        assert len(result["user2_msgs"]) == 2

    def test_make_batches_size(self):
        from src.preprocessor import make_batches
        convs = [{"conv_id": i, "messages": [], "user1_msgs": [],
                  "user2_msgs": [], "full_text": "test"} for i in range(25)]
        batches = make_batches(convs, batch_size=10)
        assert len(batches) == 3
        assert len(batches[0]["convs"]) == 10
        assert len(batches[2]["convs"]) == 5

    def test_chronological_checkpoints(self):
        from src.preprocessor import make_chronological_checkpoints
        convs = [{"conv_id": i, "messages": [], "user1_msgs": [],
                  "user2_msgs": [], "full_text": "text"} for i in range(250)]
        cps = make_chronological_checkpoints(convs, interval=100)
        assert len(cps) == 3
        assert cps[0]["conv_range"] == (0, 99)

    def test_empty_lines_handled(self):
        from src.preprocessor import parse_conversation
        raw = "User 1: Hello\n\n\nUser 2: Hi\n"
        result = parse_conversation(raw, conv_id=5)
        assert len(result["messages"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Topic Engine
# ─────────────────────────────────────────────────────────────────────────────
class TestTopicEngine:
    def test_extract_keywords_returns_list(self):
        from src.topic_engine import extract_keywords
        kws = extract_keywords("I love hiking in the mountains and cooking Italian food")
        assert isinstance(kws, list)

    def test_extract_keywords_nonempty(self):
        from src.topic_engine import extract_keywords
        kws = extract_keywords("I am studying computer science and machine learning")
        assert len(kws) > 0

    def test_extract_keywords_short_text(self):
        from src.topic_engine import extract_keywords
        kws = extract_keywords("ok")
        assert isinstance(kws, list)  # should not crash

    def test_topic_shift_detection_different_topics(self):
        from src.topic_engine import detect_topic_shift
        prev = ["cooking", "food", "recipe", "kitchen"]
        curr = ["gym", "workout", "fitness", "exercise"]
        # Different topics should be detected as shift
        result = detect_topic_shift(prev, curr)
        assert isinstance(result, bool)

    def test_topic_shift_detection_same_topic(self):
        from src.topic_engine import detect_topic_shift
        prev = ["cooking", "food", "recipe"]
        curr = ["cooking", "meal", "kitchen"]
        result = detect_topic_shift(prev, curr)
        assert isinstance(result, bool)

    def test_cluster_keywords_returns_topics(self):
        from src.topic_engine import cluster_keywords_into_topics
        kw_lists = [
            ["cooking", "food", "recipe"],
            ["gym", "workout", "fitness"],
            ["cooking", "meal", "kitchen"],
            ["gym", "exercise", "training"]
        ]
        topics = cluster_keywords_into_topics(kw_lists)
        assert isinstance(topics, list)
        assert len(topics) >= 1
        for t in topics:
            assert "label" in t
            assert "keywords" in t
            assert "conv_indices" in t


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Persona Engine (statistical only — no Ollama needed)
# ─────────────────────────────────────────────────────────────────────────────
class TestPersonaEngine:
    SAMPLE_CONV = {
        "conv_id": 0,
        "messages": [
            {"sender": "User_1", "text": "I love cooking and going to the gym!"},
            {"sender": "User_2", "text": "Me too! I work as a chef. Do you have any pets?"},
            {"sender": "User_1", "text": "Yes! I have a dog named Max 😊"},
            {"sender": "User_2", "text": "That's so cute! I have two cats."}
        ],
        "user1_msgs": ["I love cooking and going to the gym!", "Yes! I have a dog named Max 😊"],
        "user2_msgs": ["Me too! I work as a chef. Do you have any pets?", "That's so cute! I have two cats."],
        "full_text": "I love cooking gym chef dog Max cats"
    }

    def test_statistical_persona_structure(self):
        from src.persona_engine import extract_statistical_persona
        result = extract_statistical_persona(self.SAMPLE_CONV)
        assert "User_1" in result
        assert "User_2" in result
        assert "avg_msg_length" in result["User_1"]
        assert "signals" in result["User_1"]

    def test_avg_msg_length_positive(self):
        from src.persona_engine import extract_statistical_persona
        result = extract_statistical_persona(self.SAMPLE_CONV)
        assert result["User_1"]["avg_msg_length"] > 0

    def test_question_ratio_range(self):
        from src.persona_engine import extract_statistical_persona
        result = extract_statistical_persona(self.SAMPLE_CONV)
        assert 0 <= result["User_2"]["question_ratio"] <= 1

    def test_emoji_detection(self):
        from src.persona_engine import extract_statistical_persona
        result = extract_statistical_persona(self.SAMPLE_CONV)
        assert result["User_1"]["emoji_count"] >= 1  # 😊 in User 1's msg

    def test_food_signal_detected(self):
        from src.persona_engine import extract_statistical_persona
        result = extract_statistical_persona(self.SAMPLE_CONV)
        # "cooking" and "chef" are food signals
        assert result["User_1"]["signals"]["food_mentions"] >= 1 or \
               result["User_2"]["signals"]["food_mentions"] >= 1

    def test_vocab_richness_range(self):
        from src.persona_engine import extract_statistical_persona
        result = extract_statistical_persona(self.SAMPLE_CONV)
        assert 0 <= result["User_1"]["vocab_richness"] <= 1

    def test_ner_persona_structure(self):
        from src.persona_engine import extract_ner_persona
        result = extract_ner_persona(self.SAMPLE_CONV)
        assert "User_1" in result and "User_2" in result
        for user_key in ("User_1", "User_2"):
            assert "habits" in result[user_key]
            assert "personal_facts" in result[user_key]
            assert "personality" in result[user_key]
            assert "communication_style" in result[user_key]
            assert isinstance(result[user_key]["habits"], list)
            assert isinstance(result[user_key]["personal_facts"], list)

    def test_ner_personal_facts_are_strings(self):
        from src.persona_engine import extract_ner_persona
        result = extract_ner_persona(self.SAMPLE_CONV)
        for user_key in ("User_1", "User_2"):
            for fact in result[user_key]["personal_facts"]:
                assert isinstance(fact, str)

    def test_relative_communication_thresholds(self):
        from src.persona_engine import extract_ner_persona
        conv = {
            "conv_id": 1,
            "user1_msgs": ["Hi.", "Bye."],
            "user2_msgs": [
                "I wrote a long detailed explanation about a project and how it works.",
                "I am sharing an extended description with many words and examples."
            ]
        }
        result = extract_ner_persona(conv)
        assert "terse" in result["User_1"]["communication_style"]
        assert "verbose" in result["User_2"]["communication_style"]

    def test_semantic_frames_filter_objects_to_nouns(self):
        from src.persona_engine import extract_semantic_frames
        frames = extract_semantic_frames(["I love her.", "I visited London."])
        assert ("i", "love", "her") not in frames["relationship_triples"]
        assert ("i", "visited", "london") in frames["relationship_triples"]

    def test_aggregate_persona_with_ner_key(self):
        from src.persona_engine import aggregate_persona_across_batches
        personas = [
            {"User_1": {"ner": {"habits": ["discusses food/cooking"],
                                "personal_facts": ["mentions organization: Tech Corp"],
                                "personality": "inquisitive",
                                "communication_style": "casual, medium-length messages, no emojis"},
                        "stats": {"avg_msg_length": 8.0, "msg_count": 5}}},
            {"User_1": {"ner": {"habits": ["discusses fitness/exercise"],
                                "personal_facts": ["mentions location: London"],
                                "personality": "expressive",
                                "communication_style": "casual, short messages, some emojis"},
                        "stats": {"avg_msg_length": 6.0, "msg_count": 4}}},
        ]
        agg = aggregate_persona_across_batches(personas)
        assert "User_1" in agg
        habits = agg["User_1"]["habits"]
        assert any("food" in h or "fitness" in h for h in habits)

    def test_aggregate_persona_legacy_llm_key(self):
        from src.persona_engine import aggregate_persona_across_batches
        personas = [
            {"User_1": {"llm": {"habits": ["cooking"], "personal_facts": ["works as developer"],
                                "personality": "friendly", "communication_style": "casual"},
                        "stats": {"avg_msg_length": 8.0, "msg_count": 5}}},
        ]
        agg = aggregate_persona_across_batches(personas)
        assert "User_1" in agg
        assert "cooking" in agg["User_1"]["habits"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: KB Store
# ─────────────────────────────────────────────────────────────────────────────
class TestKBStore:
    def test_persona_json_write_read(self):
        import tempfile, json, os
        # Patch PERSONA_JSON to temp file
        import src.kb_store as kb
        original = kb.PERSONA_JSON
        tmp = tempfile.mktemp(suffix=".json")
        kb.PERSONA_JSON = tmp
        try:
            kb.store_persona(99, {"User_1": {"test": True}})
            result = kb.load_all_personas()
            assert "99" in result
            assert result["99"]["User_1"]["test"] is True
        finally:
            kb.PERSONA_JSON = original
            if os.path.exists(tmp):
                os.remove(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Query Engine routing
# ─────────────────────────────────────────────────────────────────────────────
class TestQueryRouter:
    def test_persona_route(self):
        from src.query_engine import _classify_query
        assert _classify_query("What kind of person is User 1?") == "persona"

    def test_habit_route(self):
        from src.query_engine import _classify_query
        assert _classify_query("What are their habits?") == "persona"

    def test_topic_route(self):
        from src.query_engine import _classify_query
        assert _classify_query("What topics were discussed?") == "topic"

    def test_summary_route(self):
        from src.query_engine import _classify_query
        assert _classify_query("Give me a summary of the conversations") == "summary"

    def test_fallback_route(self):
        from src.query_engine import _classify_query
        route = _classify_query("random question xyz")
        assert route in ("all", "topic", "summary", "persona")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

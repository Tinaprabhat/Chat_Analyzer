"""
intent_classifier.py
────────────────────
Offline Intent Classifier — Part 2, Task 2

Reads:   src/intent_model/   (fine-tuned model folder, copied from Colab)
Outputs: intent label + confidence score per user message

Classes: reminder | emotional-support | action-item | small-talk | unknown

Constraints met:
  ✅ No OpenAI / Gemini / any API calls
  ✅ Fully offline after model is copied from Colab
  ✅ CPU-only (Lenovo ThinkBook)
  ✅ <200ms per message (DistilBERT CPU ~8–15ms)
  ✅ Model < 260MB on disk (quantize to ~65MB if needed)

Integration:
  from intent_classifier import IntentClassifier
  clf = IntentClassifier()               # load once at startup
  label, score = clf.classify(message)  # call per query
"""

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_DIR   = BASE_DIR / "src" / "intent_model"

# ─────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────
INTENT_LABELS = [
    "reminder",
    "emotional-support",
    "action-item",
    "small-talk",
    "unknown",
]

# Confidence below this threshold → override to "unknown"
DEFAULT_THRESHOLD = 0.55

# ─────────────────────────────────────────────
# INTENT CLASSIFIER
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    Wraps a fine-tuned DistilBERT (from Falconsai base) loaded locally.
    Loaded once at startup. All inference is offline, CPU-only.

    Usage:
        clf = IntentClassifier()
        label, score = clf.classify("remind me to call mom")
        # → ("reminder", 0.94)
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold  = threshold
        self.model_dir  = model_dir
        self.pipeline   = None
        self._load_model()

    def _load_model(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Intent model not found at {self.model_dir}.\n"
                f"Run intent_finetune_colab.py on Google Colab, "
                f"download the zip, and unzip into src/intent_model/"
            )

        try:
            from transformers import pipeline as hf_pipeline
            log.info(f"Loading intent model from {self.model_dir} ...")
            t0 = time.time()

            self.pipeline = hf_pipeline(
                "text-classification",
                model=str(self.model_dir),
                tokenizer=str(self.model_dir),
                device=-1,          # CPU
                truncation=True,
                max_length=128,
            )

            elapsed = (time.time() - t0) * 1000
            log.info(f"Intent model loaded in {elapsed:.0f}ms")

        except Exception as e:
            log.error(f"Failed to load intent model: {e}")
            raise

    def classify(
        self,
        message: str,
        threshold: float = None,
    ) -> tuple[str, float]:
        """
        Classify a single user message.

        Args:
            message:   raw user input string
            threshold: confidence cutoff; below this → "unknown"
                       (defaults to self.threshold = 0.55)

        Returns:
            (intent_label, confidence_score)
            intent_label: one of INTENT_LABELS
            confidence_score: float 0–1
        """
        if threshold is None:
            threshold = self.threshold

        if not message or not message.strip():
            return "unknown", 0.0

        t0 = time.time()

        try:
            result = self.pipeline(message.strip())[0]
            label  = result["label"]
            score  = float(result["score"])
        except Exception as e:
            log.warning(f"Intent classification failed: {e}")
            return "unknown", 0.0

        elapsed = (time.time() - t0) * 1000

        # Low confidence → unknown
        if score < threshold:
            log.debug(f"[intent] LOW CONF {score:.3f} → unknown | {elapsed:.1f}ms | '{message[:60]}'")
            return "unknown", score

        log.debug(f"[intent] {label} ({score:.3f}) | {elapsed:.1f}ms | '{message[:60]}'")
        return label, score

    def classify_batch(
        self,
        messages: list[str],
        threshold: float = None,
    ) -> list[tuple[str, float]]:
        """
        Classify a list of messages efficiently in one batch.
        Useful for bulk classification of conversation rows.

        Returns:
            List of (intent_label, confidence_score) tuples
        """
        if threshold is None:
            threshold = self.threshold

        if not messages:
            return []

        clean = [m.strip() if m and m.strip() else "" for m in messages]

        try:
            results = self.pipeline(
                [m if m else "." for m in clean],  # pipeline rejects empty strings
                batch_size=32,
            )
        except Exception as e:
            log.warning(f"Batch classification failed: {e}")
            return [("unknown", 0.0)] * len(messages)

        output = []
        for msg, result in zip(clean, results):
            if not msg:
                output.append(("unknown", 0.0))
                continue
            label = result["label"]
            score = float(result["score"])
            if score < threshold:
                output.append(("unknown", score))
            else:
                output.append((label, score))

        return output


# ─────────────────────────────────────────────
# INTENT → QUERY ROUTER MAPPING
# ─────────────────────────────────────────────

INTENT_ROUTE_MAP = {
    "reminder":         "persona",      # check persona KB for habits/patterns
    "emotional-support":"summary",      # pull emotional context from summaries
    "action-item":      "topic",        # search topic KB for relevant context
    "small-talk":       "summary",      # lightweight summary context
    "unknown":          "all",          # full KB search as fallback
}

def intent_to_route(intent_label: str) -> str:
    """
    Map an intent label to a query route key.
    Matches the existing query_engine.py router vocabulary:
      "persona" | "topic" | "summary" | "all"
    """
    return INTENT_ROUTE_MAP.get(intent_label, "all")


# ─────────────────────────────────────────────
# STANDALONE BENCHMARK (run directly to test)
# ─────────────────────────────────────────────

def _benchmark():
    """Quick benchmark: accuracy + latency on test messages."""
    TEST_CASES = [
        # (message, expected_label)
        ("remind me to call mom tonight",             "reminder"),
        ("set a reminder for my doctor appointment",  "reminder"),
        ("don't forget the team meeting",             "reminder"),
        ("I'm feeling really anxious lately",         "emotional-support"),
        ("nobody understands what I'm going through", "emotional-support"),
        ("I just needed someone to talk to",          "emotional-support"),
        ("we need to fix this bug before release",    "action-item"),
        ("let's finalize the project timeline",       "action-item"),
        ("can you handle the code review today",      "action-item"),
        ("haha that's so funny",                      "small-talk"),
        ("what did you do last weekend",              "small-talk"),
        ("guess what happened at work today",         "small-talk"),
        ("ok",                                        "unknown"),
        ("hmm",                                       "unknown"),
        ("...",                                       "unknown"),
        # Ambiguous — should still generalize
        ("can you alert me about the presentation",   "reminder"),
        ("I feel like I'm falling apart",             "emotional-support"),
        ("let's get this shipped before the demo",    "action-item"),
        ("lmao okay that's wild",                     "small-talk"),
        ("I've been really down lately",              "emotional-support"),
    ]

    print("\n═══ Intent Classifier Benchmark ═══\n")
    clf = IntentClassifier()

    correct = 0
    latencies = []

    for msg, expected in TEST_CASES:
        t0 = time.time()
        pred_label, score = clf.classify(msg)
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)

        match = "✅" if pred_label == expected else "❌"
        if pred_label == expected:
            correct += 1

        print(
            f"  {match} [{pred_label:20s} | {score:.3f}] "
            f"({elapsed:.1f}ms)  →  {msg[:55]}"
        )

    n = len(TEST_CASES)
    print(f"\n── Results ──")
    print(f"  Accuracy    : {correct}/{n} = {correct/n*100:.1f}%")
    print(f"  Avg latency : {sum(latencies)/len(latencies):.1f}ms")
    print(f"  Max latency : {max(latencies):.1f}ms")
    print(f"  Min latency : {min(latencies):.1f}ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _benchmark()
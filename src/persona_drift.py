"""
persona_drift.py
────────────────
Adaptive Persona Engine — Part 2 of Chat Persona RAG

Reads:  data/persona_kb.json  (built by V1 pipeline)
Writes: data/persona_drift.json

Pipeline:
  1. Load persona_kb.json → flatten to per-row signals
  2. Build personalized baseline (first 20 rows per user)
  3. Normalize signals → Z-score deviation matrix
  4. Cosine similarity series (row-to-row)
  5. PELT changepoint detection on deviation matrix
  6. Tone labeling per segment
  7. Trigger extraction at each changepoint (events + NER + topic)
  8. Drift magnitude scoring (cosine between segment means)
  9. Write persona_drift.json

Dependencies:
  pip install ruptures numpy scikit-learn
  (textblob, spacy already in requirements.txt from V1)
"""

import json
import os
import logging
import numpy as np
from pathlib import Path
from typing import Any

# ruptures for PELT changepoint detection
try:
    import ruptures as rpt
except ImportError:
    raise ImportError("Install ruptures: pip install ruptures")

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PERSONA_KB_PATH = BASE_DIR / "data" / "persona_kb.json"
DRIFT_OUT_PATH  = BASE_DIR / "data" / "persona_drift.json"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASELINE_ROWS       = 20      # rows used to build personalized baseline
PELT_PENALTY        = 15      # higher = fewer changepoints (less sensitive)
COSINE_DROP_THRESH  = 0.75    # below this = candidate sudden drift
MIN_SEGMENT_LEN     = 5       # PELT min segment length (rows)
USERS               = ["User_1", "User_2"]

# ─────────────────────────────────────────────
# STEP 1 — LOAD + FLATTEN persona_kb.json
# ─────────────────────────────────────────────

def load_persona_kb(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"persona_kb.json not found at {path}. Run V1 pipeline first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_to_rows(persona_kb: dict, user: str) -> list[dict]:
    """
    persona_kb structure:
      { batch_id: { conv_id: { User_1: { stats, ner, events, preferences, traits }, User_2: {...} } } }

    Returns list of dicts, one per conversation row, in order.
    Each dict contains the raw signals for that user at that row.
    """
    rows = []
    for batch_id in sorted(persona_kb.keys()):
        batch = persona_kb[batch_id]
        for conv_id in sorted(batch.keys()):
            conv = batch[conv_id]
            user_data = conv.get(user, {})
            if not user_data:
                continue
            row = {
                "batch_id":  batch_id,
                "conv_id":   conv_id,
                "stats":     user_data.get("stats", {}),
                "events":    user_data.get("events", []),
                "ner":       user_data.get("ner", {}),
                "traits":    user_data.get("traits", []),
                "preferences": user_data.get("preferences", []),
            }
            rows.append(row)
    log.info(f"{user}: {len(rows)} rows loaded from persona_kb")
    return rows


# ─────────────────────────────────────────────
# STEP 2 — EXTRACT 7-SIGNAL FEATURE VECTOR
# ─────────────────────────────────────────────

def _parse_sentiment_from_events(events: list) -> tuple[float, float]:
    """
    V1 stores sentiment inside events as:
      [{"type": "emotion", "value": "positive"}, ...]

    Returns (polarity, negativity) as floats.
      positive → polarity +0.6, negativity 0.1
      negative → polarity -0.6, negativity 0.8
      neutral  → polarity  0.0, negativity 0.2

    Multiple emotion events → averaged.
    """
    EMOTION_MAP = {
        "positive":  ( 0.6,  0.1),
        "negative":  (-0.6,  0.8),
        "neutral":   ( 0.0,  0.2),
        "frustrated":(-0.7,  0.9),
        "excited":   ( 0.8,  0.05),
        "anxious":   (-0.3,  0.6),
        "happy":     ( 0.75, 0.05),
        "sad":       (-0.5,  0.6),
    }
    polarities, negativities = [], []
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "emotion":
            val = str(e.get("value", "neutral")).lower()
            p, n = EMOTION_MAP.get(val, (0.0, 0.2))
            polarities.append(p)
            negativities.append(n)

    if not polarities:
        return 0.0, 0.2   # default: mild neutral negativity, not 0

    return float(np.mean(polarities)), float(np.mean(negativities))


def _parse_top_trait_confidence(traits: list) -> float:
    """
    V1 stores traits as:
      [{"trait": "positive_outlook", "confidence": 0.69, "evidence": [...]}]

    Returns confidence of the highest-confidence trait (0–1).
    Used as s7 signal (trait strength / conviction signal).
    """
    if not traits or not isinstance(traits, list):
        return 0.0
    confidences = [
        float(t.get("confidence", 0))
        for t in traits
        if isinstance(t, dict)
    ]
    return max(confidences) if confidences else 0.0


def extract_feature_vector(row: dict) -> np.ndarray:
    """
    7 signals mapped to ACTUAL V1 persona_kb.json structure.

    s1  avg_msg_length     → engagement depth        (stats)
    s2  question_ratio     → curiosity / confusion   (stats)
    s3  vocab_richness     → mental energy           (stats)
    s4  emoji_count        → expressiveness          (stats, clipped at 10)
    s5  polarity           → pos/neg sentiment       (derived from events[type=emotion])
    s6  negativity         → frustration signal      (derived from events[type=emotion])
    s7  trait_confidence   → conviction / certainty  (traits[].confidence max)

    NOTE: exclamation_ratio removed — not present in V1 stats.
          Replaced by trait_confidence which captures behavioral certainty.
    """
    stats  = row.get("stats",  {})
    events = row.get("events", [])
    traits = row.get("traits", [])

    s1 = float(stats.get("avg_msg_length", 0)) / 200.0
    s2 = float(stats.get("question_ratio", 0))
    s3 = float(stats.get("vocab_richness", 0))
    s4 = min(float(stats.get("emoji_count", 0)), 10) / 10.0
    s5, s6 = _parse_sentiment_from_events(events)
    s7 = _parse_top_trait_confidence(traits)

    return np.array([s1, s2, s3, s4, s5, s6, s7], dtype=float)


def build_feature_matrix(rows: list[dict]) -> np.ndarray:
    """Returns matrix of shape (N, 7)."""
    matrix = np.array([extract_feature_vector(r) for r in rows])
    log.info(f"Feature matrix shape: {matrix.shape}")
    return matrix


# ─────────────────────────────────────────────
# STEP 3 — PERSONALIZED BASELINE + Z-SCORE
# ─────────────────────────────────────────────

def compute_baseline(matrix: np.ndarray, n: int = BASELINE_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns mean and std of first n rows.
    std is clipped at 1e-6 to avoid divide-by-zero on flat signals.
    """
    baseline = matrix[:n]
    mean = baseline.mean(axis=0)
    std  = baseline.std(axis=0)
    std  = np.clip(std, 1e-6, None)
    return mean, std


def normalize_to_zscore(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Returns deviation matrix: how many std deviations from personal baseline."""
    return (matrix - mean) / std


# ─────────────────────────────────────────────
# STEP 4 — COSINE SIMILARITY SERIES
# ─────────────────────────────────────────────

def compute_cosine_series(matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between consecutive rows.
    Returns array of shape (N-1,).
    sim[t] = cosine(row_t, row_t+1)

    Low value = sudden behavioral shift between two rows.
    """
    sims = []
    for t in range(len(matrix) - 1):
        a = matrix[t].reshape(1, -1)
        b = matrix[t + 1].reshape(1, -1)
        sim = cosine_similarity(a, b)[0][0]
        sims.append(float(sim))
    return np.array(sims)


def find_sudden_drift_candidates(cosine_series: np.ndarray, threshold: float = COSINE_DROP_THRESH) -> list[int]:
    """
    Rows where cosine dropped sharply = sudden drift candidates.
    Returns 1-indexed row numbers (matching matrix rows, not cosine series index).
    """
    candidates = [i + 1 for i, v in enumerate(cosine_series) if v < threshold]
    log.info(f"Cosine sudden drift candidates: {len(candidates)} rows below {threshold}")
    return candidates


# ─────────────────────────────────────────────
# STEP 5 — PELT CHANGEPOINT DETECTION
# ─────────────────────────────────────────────

def detect_changepoints(deviation_matrix: np.ndarray, penalty: int = PELT_PENALTY) -> list[int]:
    """
    PELT on the full 7-signal deviation matrix.
    Returns list of row indices where sustained pattern shift occurred.
    Ruptures returns end-of-segment indices; we convert to start-of-drift.
    """
    if len(deviation_matrix) < MIN_SEGMENT_LEN * 2:
        log.warning("Too few rows for PELT. Skipping.")
        return []

    algo = rpt.Pelt(model="rbf", min_size=MIN_SEGMENT_LEN).fit(deviation_matrix)
    result = algo.predict(pen=penalty)

    # result[-1] is always len(signal) — remove it (not a real changepoint)
    changepoints = [cp for cp in result if cp < len(deviation_matrix)]
    log.info(f"PELT detected {len(changepoints)} changepoints: {changepoints}")
    return changepoints


def merge_changepoints(pelt_cps: list[int], cosine_candidates: list[int], n_rows: int) -> list[int]:
    """
    Union of PELT changepoints + cosine sudden drift candidates.
    Deduplicate and sort. Remove duplicates within 3-row window (noise).
    """
    combined = sorted(set(pelt_cps + cosine_candidates))
    deduped = []
    last = -999
    for cp in combined:
        if cp - last > 3:
            deduped.append(cp)
            last = cp
    log.info(f"Merged changepoints (PELT + cosine): {len(deduped)}")
    return deduped


# ─────────────────────────────────────────────
# STEP 6 — TONE LABELING
# ─────────────────────────────────────────────

def label_tone(seg_mean: np.ndarray) -> str:
    """
    Maps mean signal vector of a segment to a human-readable tone label.
    Signals: [avg_len, q_ratio, vocab, emoji, polarity, negativity, exclamation]
    """
    avg_len     = seg_mean[0]
    q_ratio     = seg_mean[1]
    vocab       = seg_mean[2]
    emoji       = seg_mean[3]
    polarity    = seg_mean[4]
    negativity  = seg_mean[5]
    exclamation = seg_mean[6]

    # Priority order: strongest signals first
    if polarity < -0.25 and negativity > 0.35:
        if exclamation > 0.2:
            return "frustrated & reactive"
        return "frustrated & withdrawn"

    if polarity > 0.3 and emoji > 0.3:
        return "playful & expressive"

    if polarity > 0.3 and q_ratio > 0.35:
        return "curious & engaged"

    if polarity > 0.3 and avg_len > 0.5 and vocab > 0.5:
        return "thoughtful & positive"

    if q_ratio > 0.4 and vocab > 0.5:
        return "curious & analytical"

    if avg_len < 0.25 and q_ratio < 0.15 and polarity < 0.1:
        return "withdrawn & quiet"

    if exclamation > 0.25 and polarity > 0.1:
        return "energetic & expressive"

    if avg_len > 0.6 and vocab > 0.6:
        return "formal & detailed"

    if polarity >= -0.1 and polarity <= 0.1:
        return "neutral & casual"

    return "balanced"


# ─────────────────────────────────────────────
# STEP 7 — TRIGGER EXTRACTION
# ─────────────────────────────────────────────

def extract_trigger(row: dict) -> dict:
    """
    V1 actual structures:

    ner = {
      "habits":        ["discusses fitness/exercise", "discusses learning"],
      "personal_facts":["mentions location: Impala"],
      "personality":   "expressive",
      "communication_style": "casual, medium-length messages, no emojis",
      "preferences":   []
    }

    events = [
      {"type": "emotion",   "value": "positive"},
      {"type": "behavior",  "value": "active_learning"},
      {"type": "goal",      "value": "career_growth"},
      ...
    ]

    traits = [
      {"trait": "positive_outlook", "confidence": 0.69, "evidence": [...]}
    ]
    """
    events = row.get("events", [])
    ner    = row.get("ner",    {})
    traits = row.get("traits", [])

    # ── Dominant non-emotion event type ──
    # emotion events are background noise; look for behavior/goal/interest first
    non_emotion = [e for e in events if isinstance(e, dict) and e.get("type") != "emotion"]
    all_events  = [e for e in events if isinstance(e, dict)]

    dominant_event = None
    if non_emotion:
        from collections import Counter
        type_counts = Counter(e.get("type", "") for e in non_emotion)
        dominant_event = type_counts.most_common(1)[0][0]
    elif all_events:
        from collections import Counter
        type_counts = Counter(e.get("type", "") for e in all_events)
        dominant_event = type_counts.most_common(1)[0][0]

    # ── Named entities from NER dict ──
    # Extract meaningful strings from habits + personal_facts
    raw_entities = []
    if isinstance(ner, dict):
        for item in ner.get("habits", []):
            if isinstance(item, str) and len(item) > 3:
                # extract key noun: "discusses fitness/exercise" → "fitness"
                words = item.replace("/", " ").split()
                content = [w for w in words if w.lower() not in
                           {"discusses", "mentions", "talks", "about", "location:", "the", "a", "an"}]
                if content:
                    raw_entities.append(content[0])

        for item in ner.get("personal_facts", []):
            if isinstance(item, str):
                # "mentions location: Impala" → "Impala"
                parts = item.split(":")
                if len(parts) > 1:
                    raw_entities.append(parts[-1].strip())
                else:
                    raw_entities.append(item.strip())

    entities = [e for e in raw_entities if len(e) > 2][:3]

    # ── Top trait ──
    top_trait = None
    if isinstance(traits, list) and traits:
        best = max(traits, key=lambda t: float(t.get("confidence", 0)) if isinstance(t, dict) else 0)
        top_trait = best.get("trait") if isinstance(best, dict) else None

    # ── Personality / communication style as context ──
    personality    = ner.get("personality", None) if isinstance(ner, dict) else None
    comm_style     = ner.get("communication_style", None) if isinstance(ner, dict) else None

    if not dominant_event and not entities:
        return {
            "type":        "organic_drift",
            "entity":      None,
            "all_entities": [],
            "trait_shift": top_trait,
            "personality": personality,
        }

    return {
        "type":         dominant_event or "unknown",
        "entity":       entities[0] if entities else None,
        "all_entities": entities,
        "trait_shift":  top_trait,
        "personality":  personality,
        "comm_style":   comm_style,
    }


# ─────────────────────────────────────────────
# STEP 8 — DRIFT MAGNITUDE (cosine between segments)
# ─────────────────────────────────────────────

def drift_magnitude(before_rows: np.ndarray, after_rows: np.ndarray) -> float:
    """
    Cosine distance between mean of segment before vs after changepoint.
    0.0 = no drift, 1.0 = complete behavioral reversal.
    """
    if len(before_rows) == 0 or len(after_rows) == 0:
        return 0.0
    mean_before = before_rows.mean(axis=0).reshape(1, -1)
    mean_after  = after_rows.mean(axis=0).reshape(1, -1)
    sim = cosine_similarity(mean_before, mean_after)[0][0]
    return round(float(1.0 - sim), 4)


# ─────────────────────────────────────────────
# STEP 9 — MAIN DRIFT PIPELINE PER USER
# ─────────────────────────────────────────────

def run_drift_pipeline(rows: list[dict], user: str) -> dict:
    """
    Full drift detection pipeline for one user.
    Returns structured drift result dict.
    """
    if len(rows) < BASELINE_ROWS + MIN_SEGMENT_LEN:
        log.warning(f"{user}: Not enough rows ({len(rows)}) for drift analysis.")
        return {"error": "insufficient_data", "rows_found": len(rows)}

    # Build feature matrix
    matrix = build_feature_matrix(rows)

    # Personalized baseline + normalize
    mean, std = compute_baseline(matrix, BASELINE_ROWS)
    deviation_matrix = normalize_to_zscore(matrix, mean, std)

    # Cosine series on RAW matrix (not z-score — we want behavioral similarity)
    cosine_series = compute_cosine_series(matrix)
    sudden_candidates = find_sudden_drift_candidates(cosine_series)

    # PELT on deviation matrix
    pelt_cps = detect_changepoints(deviation_matrix)

    # Merge
    all_changepoints = merge_changepoints(pelt_cps, sudden_candidates, len(rows))

    # Build segments
    boundaries = [0] + all_changepoints + [len(rows)]
    segments = []

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end   = boundaries[i + 1]
        seg_rows  = rows[seg_start:seg_end]
        seg_matrix = matrix[seg_start:seg_end]

        seg_mean = seg_matrix.mean(axis=0)
        tone = label_tone(seg_mean)

        # Trigger: look at the first row of this segment (drift start)
        trigger = extract_trigger(seg_rows[0]) if seg_rows else {}

        # Drift magnitude vs previous segment
        mag = 0.0
        if i > 0:
            prev_matrix = matrix[boundaries[i - 1]:boundaries[i]]
            mag = drift_magnitude(prev_matrix, seg_matrix)

        segments.append({
            "segment":         i + 1,
            "row_start":       seg_start + 1,   # 1-indexed for readability
            "row_end":         seg_end,
            "row_count":       seg_end - seg_start,
            "tone":            tone,
            "avg_polarity":    round(float(seg_mean[4]), 3),
            "avg_negativity":  round(float(seg_mean[5]), 3),
            "drift_magnitude": mag,
            "trigger":         trigger,
        })

    # Summary stats
    tones = [s["tone"] for s in segments]
    from collections import Counter
    dominant_tone = Counter(tones).most_common(1)[0][0]

    # Cosine sudden drift rows list
    sudden_drift_rows = [{"row": r, "cosine_similarity": round(cosine_series[r - 1], 4)}
                         for r in sudden_candidates]

    return {
        "user":                  user,
        "total_rows_analyzed":   len(rows),
        "baseline_rows":         f"1–{BASELINE_ROWS}",
        "total_drifts_detected": len(all_changepoints),
        "dominant_tone":         dominant_tone,
        "pelt_changepoints":     pelt_cps,
        "cosine_sudden_drifts":  sudden_drift_rows,
        "drift_timeline":        segments,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run(persona_kb_path: Path = PERSONA_KB_PATH,
        output_path: Path = DRIFT_OUT_PATH) -> dict:
    """
    Load persona_kb.json → run drift pipeline for User_1 + User_2 → write persona_drift.json.
    Returns the full result dict.
    """
    log.info("═══ Persona Drift Engine Starting ═══")
    persona_kb = load_persona_kb(persona_kb_path)

    result = {
        "meta": {
            "source":         str(persona_kb_path),
            "pelt_penalty":   PELT_PENALTY,
            "baseline_rows":  BASELINE_ROWS,
            "cosine_threshold": COSINE_DROP_THRESH,
        }
    }

    for user in USERS:
        log.info(f"── Processing {user} ──")
        rows = flatten_to_rows(persona_kb, user)
        result[user] = run_drift_pipeline(rows, user)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"✅ persona_drift.json written → {output_path}")
    return result


if __name__ == "__main__":
    run()
from typing import List, Dict
import numpy as np
import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (EMBEDDING_MODEL, YAKE_TOP_N, YAKE_MAX_NGRAM,
                         TOPIC_SIMILARITY_THRESHOLD)
from src.logger import logger

# ── Lazy singletons ────────────────────────────────────────────────────────────
_embedder: SentenceTransformer = None

NOISE_WORDS = {
    "yeah", "yes", "no", "ok", "okay", "lol", "haha", "hey", "hi",
    "hello", "thanks", "thank", "great", "sure", "wow", "nice", "good",
    "really", "like", "know", "think", "well", "just", "also", "too",
    "that", "this", "want", "need", "got", "get", "let", "going", "going",
    "thing", "things", "way", "time", "day", "bit"
}


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _clean_text(text: str) -> str:
    """Remove noise words and short tokens."""
    tokens = text.lower().split()
    cleaned = [t for t in tokens if t not in NOISE_WORDS and len(t) > 2]
    return " ".join(cleaned)


def extract_keywords(text: str) -> List[str]:
    """Extract top keywords from text using YAKE."""
    cleaned = _clean_text(text)
    if len(cleaned.split()) < 3:
        return []
    try:
        extractor = yake.KeywordExtractor(
            lan="en",
            n=YAKE_MAX_NGRAM,
            top=YAKE_TOP_N,
            dedupLim=0.9,
        )
        results = extractor.extract_keywords(cleaned)
        return [kw for kw, _ in results]
    except Exception as e:
        logger.warning(f"YAKE failed: {e}")
        return []


def _embed_keywords(keywords: List[str]) -> np.ndarray:
    """Embed keyword list as mean of individual embeddings."""
    if not keywords:
        return np.zeros(384)
    emb = _get_embedder()
    vecs = emb.encode(keywords)
    return np.mean(vecs, axis=0)


def cluster_keywords_into_topics(all_keywords: List[List[str]]) -> List[Dict]:
    """
    Given keywords per conversation in a batch,
    cluster semantically similar ones → topics.
    Returns list of topic dicts.
    """
    if not all_keywords:
        return []

    topic_clusters = []   # list of {"label": str, "keywords": set, "conv_indices": list}

    for conv_idx, kws in enumerate(all_keywords):
        if not kws:
            continue
        vec = _embed_keywords(kws)

        # Compare to existing cluster centroids
        matched = False
        for cluster in topic_clusters:
            centroid = _embed_keywords(list(cluster["keywords"]))
            sim = cosine_similarity([vec], [centroid])[0][0]
            if sim >= TOPIC_SIMILARITY_THRESHOLD:
                cluster["keywords"].update(kws)
                cluster["conv_indices"].append(conv_idx)
                matched = True
                break

        if not matched:
            # New topic cluster
            topic_clusters.append({
                "label":        kws[0] if kws else "general",
                "keywords":     set(kws),
                "conv_indices": [conv_idx]
            })

    # Serialize sets to lists
    result = []
    for i, c in enumerate(topic_clusters):
        result.append({
            "topic_id":    i,
            "label":       c["label"],
            "keywords":    list(c["keywords"]),
            "conv_indices": c["conv_indices"]
        })
    return result


def detect_topics_in_conversation(conv: Dict) -> List[str]:
    """
    Topic detection for a single conversation.
    Returns list of keywords extracted from the conversation.
    """
    keywords = extract_keywords(conv["full_text"])
    logger.debug(f"Conv {conv['conv_id']}: extracted {len(keywords)} keywords")
    return keywords


def detect_topics_in_batch(batch: Dict) -> List[Dict]:
    """
    Full topic detection pipeline for one batch.
    Returns list of topics with keywords + conv indices.
    """
    convs = batch["convs"]
    all_keywords = [extract_keywords(c["full_text"]) for c in convs]
    topics = cluster_keywords_into_topics(all_keywords)
    logger.info(f"Batch {batch['batch_id']}: detected {len(topics)} topics")
    return topics


def detect_topic_shift(prev_keywords: List[str], curr_keywords: List[str]) -> bool:
    """
    Detect if topic shifted between two consecutive conversations.
    Used for per-conversation topic checkpoint creation.
    """
    if not prev_keywords or not curr_keywords:
        return True
    v1 = _embed_keywords(prev_keywords)
    v2 = _embed_keywords(curr_keywords)
    sim = cosine_similarity([v1], [v2])[0][0]
    return bool(sim < TOPIC_SIMILARITY_THRESHOLD)

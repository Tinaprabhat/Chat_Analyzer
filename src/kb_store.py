import json
import sqlite3
import os
import shutil
from typing import Dict, List, Any
import chromadb
from chromadb.config import Settings
from src.config import (CHROMA_DIR, PERSONA_JSON, META_DB,
                         KB_DIR, CHROMA_TOPICS_COLLECTION, CHROMA_SUMMARIES_COLLECTION,
                         CHROMA_CHECKPOINTS_COLLECTION)
from src.logger import logger

# ── ChromaDB ───────────────────────────────────────────────────────────────────
_chroma_client = None


def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _get_collection(name: str):
    return _get_chroma().get_or_create_collection(name=name)


# ── SQLite ─────────────────────────────────────────────────────────────────────
def _get_conn():
    conn = sqlite3.connect(META_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            batch_id     INTEGER PRIMARY KEY,
            conv_start   INTEGER,
            conv_end     INTEGER,
            topic_count  INTEGER,
            processed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id INTEGER PRIMARY KEY,
            conv_start    INTEGER,
            conv_end      INTEGER,
            processed_at  TEXT
        )
    """)
    conn.commit()
    return conn


# ── Store topics ───────────────────────────────────────────────────────────────
def store_topics(batch_id: int, topics: List[Dict], batch_summary: str):
    col = _get_collection(CHROMA_TOPICS_COLLECTION)
    for topic in topics:
        doc_id = f"batch_{batch_id}_topic_{topic['topic_id']}"
        doc_text = f"Topic: {topic['label']}. Keywords: {', '.join(topic['keywords'])}."
        col.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{
                "batch_id":     batch_id,
                "label":        topic["label"],
                "keywords":     ", ".join(topic["keywords"]),
                "conv_indices": str(topic["conv_indices"]),
                "summary":      batch_summary
            }]
        )
    logger.info(f"Stored {len(topics)} topics for batch {batch_id}")


# ── Store summaries ────────────────────────────────────────────────────────────
def store_summary(batch_id: int, summary: str, conv_range: tuple):
    col = _get_collection(CHROMA_SUMMARIES_COLLECTION)
    doc_id = f"summary_batch_{batch_id}"
    col.upsert(
        ids=[doc_id],
        documents=[summary],
        metadatas=[{
            "batch_id":   batch_id,
            "conv_start": conv_range[0],
            "conv_end":   conv_range[1]
        }]
    )


# ── Store checkpoint ───────────────────────────────────────────────────────────
def store_checkpoint(checkpoint_id: int, summary: str, conv_range: tuple):
    col = _get_collection(CHROMA_CHECKPOINTS_COLLECTION)
    doc_id = f"checkpoint_{checkpoint_id}"
    col.upsert(
        ids=[doc_id],
        documents=[summary],
        metadatas=[{
            "checkpoint_id": checkpoint_id,
            "conv_start":    conv_range[0],
            "conv_end":      conv_range[1]
        }]
    )
    conn = _get_conn()
    from datetime import datetime
    conn.execute(
        "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
        (checkpoint_id, conv_range[0], conv_range[1], datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


# ── Store persona ──────────────────────────────────────────────────────────────
def store_persona(batch_id: int, persona: Dict):
    os.makedirs(os.path.dirname(PERSONA_JSON), exist_ok=True)
    existing = {}
    if os.path.exists(PERSONA_JSON):
        with open(PERSONA_JSON, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[str(batch_id)] = persona
    with open(PERSONA_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ── Store batch metadata ───────────────────────────────────────────────────────
def store_batch_meta(batch_id: int, conv_range: tuple, topic_count: int):
    from datetime import datetime
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?)",
        (batch_id, conv_range[0], conv_range[1], topic_count,
         datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_topics(query: str, n_results: int = 5) -> List[Dict]:
    col = _get_collection(CHROMA_TOPICS_COLLECTION)
    try:
        results = col.query(query_texts=[query], n_results=n_results)
        out = []
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "text":     doc,
                "metadata": results["metadatas"][0][i]
            })
        return out
    except Exception as e:
        logger.error(f"Topic retrieval failed: {e}")
        return []


def retrieve_summaries(query: str, n_results: int = 5) -> List[Dict]:
    col = _get_collection(CHROMA_SUMMARIES_COLLECTION)
    try:
        results = col.query(query_texts=[query], n_results=n_results)
        out = []
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "text":     doc,
                "metadata": results["metadatas"][0][i]
            })
        return out
    except Exception as e:
        logger.error(f"Summary retrieval failed: {e}")
        return []


def retrieve_checkpoints(query: str, n_results: int = 3) -> List[Dict]:
    col = _get_collection(CHROMA_CHECKPOINTS_COLLECTION)
    try:
        results = col.query(query_texts=[query], n_results=n_results)
        out = []
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "text":     doc,
                "metadata": results["metadatas"][0][i]
            })
        return out
    except Exception as e:
        logger.error(f"Checkpoint retrieval failed: {e}")
        return []


def load_all_personas() -> Dict:
    if not os.path.exists(PERSONA_JSON):
        return {}
    with open(PERSONA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_kb() -> None:
    """Remove all KB artifacts so the knowledge base can be rebuilt from scratch."""
    global _chroma_client
    _chroma_client = None
    try:
        shutil.rmtree(KB_DIR, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to delete KB directory: {e}")
    os.makedirs(KB_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)


def is_kb_populated() -> bool:
    """Check if KB has been built already."""
    col = _get_collection(CHROMA_SUMMARIES_COLLECTION)
    try:
        return col.count() > 0
    except Exception:
        return False

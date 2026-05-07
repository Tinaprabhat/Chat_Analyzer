from tqdm import tqdm
from src.preprocessor import load_and_parse, make_batches, make_chronological_checkpoints
from src.topic_engine import detect_topics_in_batch, detect_topics_in_conversation, cluster_keywords_into_topics
from src.summarizer import summarize_batch, summarize_checkpoint, summarize_topic_segment
from src.persona_engine import extract_full_persona, aggregate_persona_across_batches
from src.kb_store import (store_topics, store_summary, store_checkpoint,
                           store_persona, store_batch_meta, is_kb_populated,
                           load_all_personas)
from src.logger import logger, log_update
import json
import os
from src.config import PERSONA_JSON, KB_DIR


def run_pipeline(csv_path: str = None, pdf_path: str = None, force_rebuild: bool = False):
    """
    Full ingestion pipeline:
    1. Load + parse CSV or PDF
    2. Batch (10 convs/batch)
    3. Per batch: topics + summary + persona → store in KB
    4. Chronological checkpoints (every 100 convs)
    5. Aggregate final persona
    
    Args:
        csv_path: Path to CSV file (optional)
        pdf_path: Path to PDF file (optional)
        force_rebuild: Force rebuilding even if KB exists
    """
    if is_kb_populated() and not force_rebuild:
        logger.info("KB already populated. Skipping pipeline. Use force_rebuild=True to re-run.")
        return

    logger.info("=" * 60)
    logger.info("PIPELINE START")

    # ── Step 1: Load ───────────────────────────────────────────────
    from src.config import CSV_PATH, PDF_PATH
    
    # Determine which path to use
    path = None
    if csv_path:
        path = csv_path
    elif pdf_path:
        path = pdf_path
    elif os.path.exists(CSV_PATH):
        path = CSV_PATH
    elif os.path.exists(PDF_PATH):
        path = PDF_PATH
    else:
        logger.error("No CSV or PDF file found")
        raise FileNotFoundError("No CSV or PDF file found in data directory")
    
    conversations = load_and_parse(path)
    log_update(f"Loaded {len(conversations)} conversations from {os.path.basename(path)}")

    # ── Step 2: Batch ──────────────────────────────────────────────
    batches = make_batches(conversations)
    log_update(f"Created {len(batches)} batches")

    # ── Step 3: Per-batch extraction ───────────────────────────────
    all_personas = []
    for batch in tqdm(batches, desc="Processing batches"):
        bid = batch["batch_id"]
        try:
            # Topics: Extract keywords per conversation, then cluster
            all_keywords = []
            for conv in batch["convs"]:
                keywords = detect_topics_in_conversation(conv)
                all_keywords.append(keywords)
            topics = cluster_keywords_into_topics(all_keywords)

            # Summary (batch-level)
            summary = summarize_batch(batch)

            # Per-topic summaries
            convs_by_topic_idx = {t["topic_id"]: [] for t in topics}
            for t in topics:
                for ci in t["conv_indices"]:
                    if ci < len(batch["convs"]):
                        convs_by_topic_idx[t["topic_id"]].append(batch["convs"][ci])

            topic_summaries = {}
            for t in topics:
                tsummary = summarize_topic_segment(
                    convs_by_topic_idx[t["topic_id"]], t["label"]
                )
                topic_summaries[t["topic_id"]] = tsummary

            # Attach topic summaries to topics
            for t in topics:
                t["topic_summary"] = topic_summaries.get(t["topic_id"], "")

            # Persona per conversation in batch
            batch_personas = {}
            for conv in batch["convs"]:
                persona = extract_full_persona(conv)
                batch_personas[conv["conv_id"]] = persona
                all_personas.append(persona)

            # Store everything
            store_topics(bid, topics, summary)
            store_summary(bid, summary, batch["conv_range"])
            store_persona(bid, batch_personas)
            store_batch_meta(bid, batch["conv_range"], len(topics))

            log_update(f"Batch {bid} processed and stored")

        except Exception as e:
            logger.error(f"Batch {bid} failed: {e}")
            continue

    # ── Step 4: Chronological checkpoints ─────────────────────────
    checkpoints = make_chronological_checkpoints(conversations)
    for cp in tqdm(checkpoints, desc="Building checkpoints"):
        try:
            cp_summary = summarize_checkpoint(cp)
            store_checkpoint(cp["checkpoint_id"], cp_summary, cp["conv_range"])
            log_update(f"Checkpoint {cp['checkpoint_id']} stored")
        except Exception as e:
            logger.error(f"Checkpoint {cp['checkpoint_id']} failed: {e}")

    # ── Step 5: Aggregate final persona ───────────────────────────
    if all_personas:
        flat_personas = []
        for p in all_personas:
            for user_key in ["User_1", "User_2"]:
                if user_key in p:
                    flat_personas.append({user_key: p[user_key]})

        aggregated = aggregate_persona_across_batches(
            [{uk: p[uk]} for p in all_personas for uk in ["User_1", "User_2"] if uk in p]
        )
        agg_path = os.path.join(KB_DIR, "aggregated_persona.json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        log_update("Aggregated persona saved")

    logger.info("PIPELINE COMPLETE")
    log_update("Full pipeline complete")

import pandas as pd
import re
from typing import List, Dict
from src.config import CSV_PATH, PDF_PATH, BATCH_SIZE, CHECKPOINT_INTERVAL
from src.logger import logger
import os


def parse_conversation(raw: str, conv_id: int) -> Dict:
    """Parse raw conversation string into structured dict."""
    messages = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(User\s*\d+):\s*(.+)$", line, re.IGNORECASE)
        if match:
            sender = match.group(1).strip().replace(" ", "_")
            text   = match.group(2).strip()
            messages.append({"sender": sender, "text": text, "msg_idx": len(messages)})
    return {
        "conv_id":  conv_id,
        "messages": messages,
        "user1_msgs": [m["text"] for m in messages if "1" in m["sender"]],
        "user2_msgs": [m["text"] for m in messages if "2" in m["sender"]],
        "full_text": " ".join(m["text"] for m in messages)
    }


def load_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logger.info(f"Extracted text from {len(reader.pages)} PDF pages")
        return text
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        return ""


def load_and_parse(data_path: str = None) -> List[Dict]:
    """Load and parse conversations from CSV or PDF.
    
    Args:
        data_path: Path to CSV or PDF file. If None, tries CSV_PATH first, then PDF_PATH.
    
    Returns:
        List of parsed conversation dicts.
    """
    # Determine which file to load
    if data_path is None:
        if os.path.exists(CSV_PATH):
            data_path = CSV_PATH
        elif os.path.exists(PDF_PATH):
            data_path = PDF_PATH
        else:
            logger.error("No CSV or PDF found at default paths")
            return []
    
    conversations = []
    
    # Handle PDF
    if data_path.lower().endswith(".pdf"):
        text = load_pdf_text(data_path)
        if not text.strip():
            logger.error("PDF is empty or unreadable")
            return []
        
        # Split text into conversations (split by double newlines or conversation markers)
        raw_convs = text.split("\n\n")
        for idx, raw_conv in enumerate(raw_convs):
            if raw_conv.strip():
                parsed = parse_conversation(raw_conv, conv_id=idx)
                if parsed["messages"]:
                    conversations.append(parsed)
        logger.info(f"Loaded {len(conversations)} conversations from PDF")
    # Handle CSV
    else:
        df = pd.read_csv(data_path)
        col = df.columns[0]
        for idx, row in df.iterrows():
            parsed = parse_conversation(str(row[col]), conv_id=idx)
            if parsed["messages"]:
                conversations.append(parsed)
        logger.info(f"Loaded {len(conversations)} conversations from CSV")
    
    return conversations


def make_batches(conversations: List[Dict], batch_size: int = BATCH_SIZE) -> List[Dict]:
    """Group conversations into batches."""
    batches = []
    for i in range(0, len(conversations), batch_size):
        chunk = conversations[i: i + batch_size]
        batches.append({
            "batch_id":   i // batch_size,
            "conv_range": (i, min(i + batch_size - 1, len(conversations) - 1)),
            "convs":      chunk
        })
    logger.info(f"Created {len(batches)} batches of size {batch_size}")
    return batches


def make_chronological_checkpoints(conversations: List[Dict],
                                   interval: int = CHECKPOINT_INTERVAL) -> List[Dict]:
    """Every `interval` conversations → one checkpoint (independent of topics)."""
    checkpoints = []
    for i in range(0, len(conversations), interval):
        chunk = conversations[i: i + interval]
        checkpoints.append({
            "checkpoint_id": i // interval,
            "conv_range":    (i, min(i + interval - 1, len(conversations) - 1)),
            "convs":         chunk,
            "full_text":     " ".join(c["full_text"] for c in chunk)
        })
    logger.info(f"Created {len(checkpoints)} chronological checkpoints (interval={interval})")
    return checkpoints

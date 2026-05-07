import nltk
from typing import Dict, List
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from src.logger import logger

# Ensure NLTK sentence tokenizer data is present
for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if _pkg != "stopwords" else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_summarizer = TextRankSummarizer()


def _textrank(text: str, n_sentences: int) -> str:
    """Extract top-n sentences from text using TextRank. Returns '' on failure."""
    text = text.strip()
    if not text:
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        sentences = _summarizer(parser.document, n_sentences)
        return " ".join(str(s) for s in sentences)
    except Exception as e:
        logger.warning(f"TextRank failed: {e}")
        return ""


def _conv_text(convs: List[Dict]) -> str:
    lines = []
    for c in convs:
        for m in c["messages"]:
            lines.append(f"{m['sender']}: {m['text']}")
    return "\n".join(lines)


def summarize_batch(batch: Dict) -> str:
    text = _conv_text(batch["convs"])
    summary = _textrank(text, n_sentences=4)
    if not summary:
        # Fallback: first message of each conversation
        sentences = [c["messages"][0]["text"] for c in batch["convs"] if c["messages"]]
        summary = " | ".join(sentences[:5])
    logger.info(f"Batch {batch['batch_id']}: summary generated ({len(summary)} chars)")
    return summary


def summarize_topic_segment(convs: List[Dict], topic_label: str) -> str:
    text = _conv_text(convs)
    summary = _textrank(text, n_sentences=3)
    if not summary:
        summary = f"Conversations discussing {topic_label}."
    return summary


def summarize_checkpoint(checkpoint: Dict) -> str:
    text = _conv_text(checkpoint["convs"][:10])
    summary = _textrank(text, n_sentences=4)
    if not summary:
        summary = (f"Checkpoint covering conversations "
                   f"{checkpoint['conv_range'][0]}-{checkpoint['conv_range'][1]}.")
    logger.info(f"Checkpoint {checkpoint['checkpoint_id']}: summary generated")
    return summary

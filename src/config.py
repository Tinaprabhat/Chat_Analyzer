import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
KB_DIR       = os.path.join(BASE_DIR, "kb")
LOG_DIR      = os.path.join(BASE_DIR, "logs")

CSV_PATH     = os.path.join(DATA_DIR, "conversations.csv")
PDF_PATH     = os.path.join(DATA_DIR, "conversations.pdf")
PERSONA_JSON = os.path.join(KB_DIR,   "persona_kb.json")
META_DB      = os.path.join(KB_DIR,   "metadata.db")
CHROMA_DIR   = os.path.join(KB_DIR,   "chromadb")
LOG_FILE     = os.path.join(LOG_DIR,  "project.log")

for d in [DATA_DIR, KB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Batching ───────────────────────────────────────────────────────────────────
BATCH_SIZE              = 100         # conversations per batch
CHECKPOINT_INTERVAL     = 100         # 100-conversation chronological checkpoint

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
GROQ_MODEL              = "llama-3.3-70b-versatile"

# ── Ollama (Local LLM Fallback) ────────────────────────────────────────────────
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── API Keys (from .env file) ──────────────────────────────────────────────────
GROQ_API_KEY            = os.getenv("GROQ_API_KEY", "")

# ── YAKE ─────────────────────────────────────────────────────────────────────
YAKE_TOP_N              = 5
YAKE_MAX_NGRAM          = 2

# ── Topic detection ───────────────────────────────────────────────────────────
TOPIC_SIMILARITY_THRESHOLD = 0.35    # below = new topic

# ── ChromaDB Collections ──────────────────────────────────────────────────────
CHROMA_TOPICS_COLLECTION     = "topics"
CHROMA_SUMMARIES_COLLECTION  = "summaries"
CHROMA_CHECKPOINTS_COLLECTION = "checkpoints"

# 🧠 Chat Persona RAG

> **Analyze any WhatsApp / chat export — understand personas, topics, and conversation patterns — without sending your data to any LLM during knowledge base construction.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatanalyzer-6nr3wskqbs5obudbmnwib9.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/LLM%20query%20tests-50%2F50%20passed-brightgreen)

---

## What It Does

Upload a `.csv` or `.pdf` of your chat export. The system:

1. **Builds a complete Knowledge Base** — entirely using classical NLP and mathematical methods (zero LLM calls)
2. **Extracts topics, summaries, and personas** per conversation batch
3. **Answers natural language queries** about the conversation using a Groq-powered LLM grounded strictly in the built KB

Your data never touches an LLM during the heavy lifting. The LLM is only invoked at query time — and even then, it is forbidden from guessing outside the retrieved context.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        INPUT                            │
│         File Uploader: .csv or .pdf (Streamlit UI)      │
│        Each row / page = 1 full conversation            │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  PREPROCESSING                          │
│  • Parse each row → split by \n                         │
│  • Label sender: User1 / User2                          │
│  • Strip noise (empty lines, artifacts)                 │
│  • Assign conversation_id (row index)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│           BATCHING  (10 convos = 1 batch)               │
│       batch_001: conv_0001 → conv_0010                  │
│       batch_002: conv_0011 → conv_0020  ...             │
│            Total batches = n_conversations / 10         │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
┌──────────▼──────────┐  ┌────────▼──────────────────────┐
│   TOPIC ENGINE      │  │      PERSONA ENGINE            │
│   (Mathematical)    │  │      (Mathematical)            │
│                     │  │                                │
│ KeyBERT             │  │  STATISTICAL (per user):       │
│ → top 5 keywords    │  │  • avg message length          │
│   per conversation  │  │  • who initiates more          │
│                     │  │  • topic diversity score       │
│ all-MiniLM-L6-v2    │  │  • vocabulary richness         │
│ → embed keywords    │  │  • question frequency          │
│ → cosine clustering │  │                                │
│ → topic labels      │  │  SpaCy NER:                    │
│ → topic shifts      │  │  • named entity extraction     │
│   per batch         │  │  • habit & fact inference      │
│                     │  │  • communication style         │
└──────────┬──────────┘  └────────┬──────────────────────┘
           │                      │
┌──────────▼──────────────────────▼──────────────────────┐
│              SUMMARIZATION ENGINE                       │
│              TextRank (Extractive NLP)                  │
│    Input: 10 conversations (1 batch)                    │
│    Output: 1 extractive summary per batch               │
│    ─────────────────────────────────────────────────   │
│    ✅ Zero LLM calls during KB construction             │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              KNOWLEDGE BASE  (3-layer)                  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ChromaDB Collection: "topics"                  │   │
│  │  doc: topic label + keywords                    │   │
│  │  metadata: batch_id, conv_ids, msg_indices      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ChromaDB Collection: "summaries"               │   │
│  │  doc: TextRank extractive summary               │   │
│  │  metadata: batch_id, conv_range                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  persona_kb.json                                │   │
│  │  { batch_id: {                                  │   │
│  │      user1: {habits, facts, style, entities},   │   │
│  │      user2: {habits, facts, style, entities},   │   │
│  │      stats: {stat engine outputs}               │   │
│  │    }                                            │   │
│  │  }                                              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  SQLite: metadata.db                            │   │
│  │  batch_id | conv_range | topic_count |          │   │
│  │  processed_at | summary_id | status             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  QUERY ENGINE                           │
│                                                         │
│  User Query (Streamlit UI)                              │
│         ↓                                               │
│  Query Router:                                          │
│  • "persona" query   → scan persona_kb.json (all batch) │
│  • "topic/event"     → ChromaDB semantic search         │
│  • "summary"         → ChromaDB summaries collection    │
│  • complex query     → both                             │
│         ↓                                               │
│  Context Builder → assembles retrieved KB chunks        │
│         ↓                                               │
│  ┌─────────────────────────────────────────┐           │
│  │  Groq API  (primary)                    │           │
│  │  llama3-70b  or  mixtral-8x7b           │           │
│  │  Prompt: "Answer ONLY from context.     │           │
│  │           No hallucination."            │           │
│  └────────────────┬────────────────────────┘           │
│                   │  (on failure / rate-limit)          │
│  ┌────────────────▼────────────────────────┐           │
│  │  Fallback: Ollama (local)               │           │
│  │  Model: qwen2.5:1.5b                    │           │
│  └─────────────────────────────────────────┘           │
│         ↓                                               │
│  Response → Streamlit UI                                │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **LLM-free KB construction** | Privacy. Your raw conversations never leave your machine during indexing. |
| **TextRank for summarization** | Extractive and deterministic. No hallucinated summaries. Reproducible. |
| **SpaCy NER for persona** | Fast, local, and entity-grounded. Habits and facts tied to real named entities. |
| **KeyBERT + cosine clustering for topics** | Semantically meaningful topics without any API call. |
| **ChromaDB** | Lightweight, embeddable vector store. No server setup required. |
| **Groq primary + Ollama fallback** | Low latency cloud inference with zero-downtime local fallback. |
| **"Answer ONLY from context" prompt** | Prevents LLM from fabricating details about real people's conversations. |

---

## Tech Stack

**NLP / ML**
- `KeyBERT` — keyword extraction
- `sentence-transformers` (all-MiniLM-L6-v2) — semantic embeddings
- `spaCy` (en_core_web_sm) — Named Entity Recognition for persona building
- `TextRank` (via `sumy` or `pytextrank`) — extractive summarization

**Storage**
- `ChromaDB` — vector store for topics and summaries
- `SQLite` — batch metadata and processing state
- `JSON` — persona knowledge base

**Inference**
- `Groq API` — primary LLM (llama3-70b / mixtral-8x7b)
- `Ollama` (qwen2.5:1.5b) — local fallback

**UI & Deployment**
- `Streamlit` — file uploader, query interface, results display
- `Streamlit Cloud` — deployment

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/<your-username>/chat-persona-rag.git
cd chat-persona-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set API keys

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key_here
```

For local fallback, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull qwen2.5:1.5b
```

### 4. Run

```bash
streamlit run app.py
```

---

## Input Format

**CSV** — each row is one conversation. Messages within a row are separated by `\n`. Example:

```
conversation
"Hey are you free tonight?\nYeah what's up?\nWanna catch a movie?"
```

**PDF** — each page is treated as one conversation block.

---

## Query Examples

Once the KB is built, you can ask:

```
Who initiates conversations more often?
What topics came up the most between batch 3 and batch 7?
Summarize what User1 talks about most.
What are User2's communication habits?
Were there any conversations about travel?
```

---

## Testing

The query engine has been tested against **50 LLM query cases** — all passed.

Tests validate:
- Correct routing (persona / topic / summary / hybrid)
- Context retrieval accuracy
- LLM non-hallucination (answers grounded in KB only)
- Fallback activation when Groq is unavailable

---

## Limitations

- Persona inference is NER-based, not deep psychological profiling. It reflects what entities appear in text, not implicit personality.
- TextRank summaries are extractive — they pull existing sentences, not paraphrase. Long or noisy conversations may produce awkward summaries.
- Topic clustering quality depends on conversation length. Very short exchanges may produce weak keywords.
- The system assumes two speakers (User1 / User2). Group chats are not currently supported.

---

## Live Demo

🔗 [chatanalyzer-6nr3wskqbs5obudbmnwib9.streamlit.app](https://chatanalyzer-6nr3wskqbs5obudbmnwib9.streamlit.app/)

---

## Author

**Tina Prabhat**
BTech Final Year | ML · AI · NLP
Paper accepted at **ICMEET 2025, London**

---

## License

MIT

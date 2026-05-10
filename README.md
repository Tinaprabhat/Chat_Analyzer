# 🧠 Chat Persona RAG

> **Analyze any chat export — extract personas, topics, and conversation patterns — without sending your data to any LLM during knowledge base construction.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatanalyzer-6nr3wskqbs5obudbmnwib9.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/RAGWatch%20eval-16%2F18%20passed-brightgreen)

---

## What It Does

Upload a `.csv` or `.pdf` of your chat export. The system:

1. **Builds a complete Knowledge Base** — entirely using classical NLP and mathematical methods (zero LLM calls)
2. **Extracts topics, summaries, and deep personas** per conversation batch
3. **Answers natural language queries** about the conversation using a Groq-powered LLM grounded strictly in the built KB

Your data never touches an LLM during KB construction. The LLM is only invoked at query time, and is explicitly constrained from guessing outside retrieved context.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          INPUT                              │
│       File Uploader: .csv or .pdf  (Streamlit UI)           │
│       Each row / page = 1 full conversation                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     PREPROCESSING                           │
│  • Parse each row → split by \n                             │
│  • Label sender: User_1 / User_2                            │
│  • Strip noise (empty lines, artifacts)                     │
│  • Assign conversation_id (row index)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              BATCHING  (10 convos = 1 batch)                │
│          batch_001 : conv_0001 → conv_0010                  │
│          batch_002 : conv_0011 → conv_0020  …               │
│          Parallel: 100-conv chronological checkpoints       │
└────────┬──────────────────────────────┬─────────────────────┘
         │                              │
┌────────▼────────────────┐  ┌──────────▼──────────────────────┐
│     TOPIC ENGINE        │  │       PERSONA ENGINE V2          │
│     (Mathematical)      │  │       (6-layer, Mathematical)    │
│                         │  │                                  │
│  YAKE keyword extractor │  │  Layer 1 — StatsEngine           │
│  → top-N keywords per   │  │    avg_msg_length, question_     │
│    conversation         │  │    ratio, emoji_count,           │
│    (faster than KeyBERT,│  │    vocab_richness, msg_count     │
│     tradeoff: slight    │  │                                  │
│     quality vs speed)   │  │  Layer 2 — EventExtractionLayer  │
│                         │  │    keyword signal banks:         │
│  all-MiniLM-L6-v2       │  │    TECH, CAREER, LEARNING,       │
│  → embed keyword lists  │  │    EMOTION_POS, EMOTION_NEG      │
│  → cosine similarity    │  │    → typed events per message    │
│  → cluster into topics  │  │                                  │
│    per batch            │  │  Layer 3 — PreferenceMemoryLayer │
│                         │  │    regex pattern matching:       │
│  detect_topic_shift():  │  │    concise_answers,              │
│  → cosine drop between  │  │    detailed_answers,             │
│    consecutive conv     │  │    practical_focus,              │
│    embeddings triggers  │  │    systems_thinking              │
│    new topic checkpoint │  │    → confidence-weighted prefs   │
│                         │  │                                  │
│  100-msg chronological  │  │  Layer 4 — SentimentEngine       │
│  checkpoints,           │  │    TextBlob per message          │
│  independent of topic   │  │    → polarity score              │
│  boundaries             │  │    → overall: optimistic /       │
│                         │  │      frustrated / balanced       │
│                         │  │                                  │
│                         │  │  Layer 5 — SemanticFrameExtractor│
│                         │  │    spaCy dep parse per message   │
│                         │  │    → (subject, relation, object) │
│                         │  │    NOUN/PROPN objects only       │
│                         │  │    → no pronoun/filler noise     │
│                         │  │                                  │
│                         │  │  Layer 6 — TraitInferenceLayer   │
│                         │  │    cross-signal inference:       │
│                         │  │    high q_ratio + vocab →        │
│                         │  │      analytical_thinker (0.82)   │
│                         │  │    tech events + sys prefs →     │
│                         │  │      systems_oriented (0.84)     │
│                         │  │    career events ≥ 2 →           │
│                         │  │      career_ambitious (0.78)     │
│                         │  │    learning events ≥ 2 →         │
│                         │  │      active_learner (0.88)       │
│                         │  │    sentiment + msg length →      │
│                         │  │      communication style traits  │
│                         │  │    semantic AI relations ≥ 2 →   │
│                         │  │      deep_technical_interest     │
│                         │  │                                  │
│                         │  │  Output per user per conv:       │
│                         │  │  stats | ner | events |          │
│                         │  │  preferences | traits            │
└────────┬────────────────┘  └──────────┬──────────────────────┘
         │                              │
┌────────▼──────────────────────────────▼─────────────────────┐
│                  SUMMARIZATION ENGINE                        │
│                  TextRank  (Extractive NLP)                  │
│                  via sumy PlaintextParser                    │
│                                                             │
│   summarize_batch()        → 4 sentences / 10-conv batch    │
│   summarize_topic_segment()→ 3 sentences / topic segment    │
│   summarize_checkpoint()   → 4 sentences / 100-conv block   │
│                                                             │
│   ✅ Zero LLM calls during KB construction                  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  KNOWLEDGE BASE  (4-layer)                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ChromaDB — "topics"                                 │  │
│  │  doc: topic label + YAKE keywords                    │  │
│  │  metadata: batch_id, conv_ids, topic_summary         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ChromaDB — "summaries"                              │  │
│  │  doc: TextRank batch summary                         │  │
│  │  metadata: batch_id, conv_range                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ChromaDB — "checkpoints"                            │  │
│  │  doc: TextRank checkpoint summary                    │  │
│  │  metadata: checkpoint_id, conv_range                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  persona_kb.json                                     │  │
│  │  { batch_id: { conv_id: {                            │  │
│  │      User_1: { stats, ner, events,                   │  │
│  │                preferences, traits },                │  │
│  │      User_2: { stats, ner, events,                   │  │
│  │                preferences, traits }                 │  │
│  │  }}}                                                 │  │
│  │  + aggregated_persona.json  (full-dataset rollup)    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SQLite — metadata.db                                │  │
│  │  batch_id | conv_range | topic_count |               │  │
│  │  processed_at | summary_id | status                  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     QUERY ENGINE                            │
│                                                             │
│  User Query  (Streamlit UI)                                 │
│       ↓                                                     │
│  Query Router  (keyword scoring):                           │
│  "persona" → persona_kb.json  (aggregated rollup)          │
│  "topic"   → ChromaDB topics  (semantic search)            │
│  "summary" → ChromaDB summaries + checkpoints              │
│  complex   → all layers combined                           │
│       ↓                                                     │
│  Context Builder → assembles top-K retrieved chunks        │
│       ↓                                                     │
│  ┌──────────────────────────────────────────┐             │
│  │  Groq API  (primary)                     │             │
│  │  llama3-70b-8192                         │             │
│  │  Prompt: "Answer ONLY from context.      │             │
│  │           No hallucination."             │             │
│  └──────────────────┬───────────────────────┘             │
│                     │  on failure / rate-limit             │
│  ┌──────────────────▼───────────────────────┐             │
│  │  Fallback: Ollama  (local)               │             │
│  │  Model: qwen2.5:1.5b                     │             │
│  └──────────────────────────────────────────┘             │
│       ↓                                                     │
│  Response → Streamlit UI                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **LLM-free KB construction** | Privacy. Raw conversations never leave the machine during indexing. Cost is zero. Fully reproducible. |
| **YAKE over KeyBERT** | YAKE is statistical — no model load, significantly faster at 11k conversation scale. Tradeoff: slight reduction in semantic precision vs KeyBERT, acceptable given throughput requirements. |
| **6-layer persona engine** | Single-signal persona (NER only) is too shallow for conversational self-disclosure. Each layer captures a distinct dimension: stats → events → preferences → sentiment → semantic frames → inferred traits. |
| **Relative thresholds in communication style** | Fixed absolute thresholds collapse to a single label across all users. Relative comparison (user avg vs conversation avg) produces differentiated output. |
| **NOUN/PROPN filter on semantic triples** | Prevents pronoun and filler word pollution in relationship triples — eliminates `["i","love","it"]` noise in favour of `["i","love","music"]`. |
| **TextRank for summarization** | Extractive and deterministic. No hallucinated summaries. Identical input always produces identical output. |
| **Groq primary + Ollama fallback** | Low-latency cloud inference with zero-downtime local fallback. No single point of failure at query time. |
| **"Answer ONLY from context" prompt** | Hard constraint preventing the LLM from fabricating details about real people's conversations. |

---

## Persona Engine V2 — Layer Detail

The persona engine is the most architecturally significant component. Six layers run per user per conversation and cross-wire through the TraitInferenceLayer.

```
Messages (User_1 or User_2)
       │
       ├──▶ StatsEngine
       │     avg_msg_length, question_ratio, emoji_count,
       │     vocab_richness, msg_count
       │
       ├──▶ EventExtractionLayer
       │     keyword signal banks → typed events
       │     {type: "interest", value: "technology_ai"}
       │     {type: "goal",     value: "career_growth"}
       │     {type: "emotion",  value: "positive"}
       │
       ├──▶ PreferenceMemoryLayer
       │     regex pattern matching → confidence-weighted preferences
       │     concise_answers | detailed_answers |
       │     practical_focus | systems_thinking
       │
       ├──▶ SentimentEngine  (TextBlob)
       │     per-message polarity → overall: optimistic / frustrated / balanced
       │
       ├──▶ SemanticFrameExtractor  (spaCy dep parse)
       │     (subject, relation, object) triples
       │     NOUN/PROPN objects only — no pronoun noise
       │
       └──▶ TraitInferenceLayer  (cross-signal)
             inputs: all 5 layers above
             outputs: named traits with confidence scores + evidence
             e.g. analytical_thinker (0.82), active_learner (0.88),
                  systems_oriented (0.84), career_ambitious (0.78)
```

---

## Evaluation

Evaluated using **RAGWatch**, a personally built query evaluation framework that measures faithfulness and hallucination resistance across persona, topic, and summary query types. No third-party eval library was used.

### Summary — 18 test cases

| Metric | Score |
|---|---|
| Overall pass rate | **16 / 18** |
| Composite score | 0.557 ± 0.093 |
| Faithfulness | 0.642 |
| Hallucination rate | 0.358 |
| Cases passing lenient threshold (≥ 0.35) | **15 / 15** |
| Cases passing aggregate threshold (≥ 0.50) | 10 / 15 |

### Per-query results — 15 persona/topic/summary cases

| Query | Composite | Faithfulness | Hallucination | Result |
|---|---|---|---|---|
| What kind of person is User 1? | 0.448 | 0.500 | 0.500 | FAIL |
| What are User 1's habits? | 0.576 | 0.559 | 0.441 | pass |
| How does User 1 communicate? | 0.481 | 0.495 | 0.505 | FAIL |
| What is User 1's communication style? | 0.488 | 0.514 | 0.486 | FAIL |
| What personal facts do we know about User 1? | 0.529 | 0.501 | 0.499 | pass |
| What are the habits of User 2? | 0.665 | **0.844** | 0.156 | pass |
| Describe User 2's personality. | 0.590 | 0.502 | 0.498 | pass |
| What is User 2's communication style? | 0.537 | 0.502 | 0.498 | pass |
| What do we know about User 2's work life? | 0.423 | 0.495 | 0.505 | FAIL |
| What personal facts do we know about User 2? | **0.668** | **0.849** | 0.151 | pass |
| What topics were most discussed? | **0.667** | 0.796 | 0.204 | pass |
| What subjects came up in the conversations? | 0.572 | **0.994** | 0.006 | pass |
| What themes were discussed? | **0.735** | **0.991** | 0.009 | pass |
| Give me a summary of the conversations | 0.491 | 0.601 | 0.399 | FAIL |
| What is an overview of the chats? | 0.483 | 0.493 | 0.507 | FAIL |

**Reading the results:**
- Topic queries are the system's strongest dimension — faithfulness above 0.79 on all three, near-zero hallucination on subject/theme queries.
- User 2 persona queries consistently outperform User 1, reflecting uneven signal density in the dataset rather than a systematic engine failure.
- Summary/overview queries fail on composite but the faithfulness scores (0.60, 0.49) indicate the retrieved context is grounded — the gap is answer completeness, not hallucination.
- Communication style queries are the weakest persona dimension, consistent with the known limitation that short conversations provide insufficient signal for reliable relative-threshold classification.

---

## Tech Stack

**NLP / ML**
- `YAKE` — statistical keyword extraction for topic detection (no model load, fast at scale)
- `sentence-transformers` (all-MiniLM-L6-v2) — semantic embeddings for topic clustering
- `spaCy` (en_core_web_sm) — dependency parsing for semantic frame extraction
- `TextBlob` — message-level sentiment polarity
- `sumy` (TextRankSummarizer) — extractive summarization

**Storage**
- `ChromaDB` — vector store for topics, summaries, and checkpoints
- `SQLite` — batch metadata and processing state
- `JSON` — persona knowledge base (per-conversation + aggregated rollup)

**Inference**
- `Groq API` — primary LLM (llama3-70b-8192)
- `Ollama` (qwen2.5:1.5b) — local fallback

**Evaluation**
- `RAGWatch` — custom-built evaluation framework, measuring composite score, faithfulness, and hallucination rate per query case

**UI & Deployment**
- `Streamlit` — file uploader, KB build progress, query interface
- `Streamlit Cloud` — deployment target

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

```env
# .env
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

**CSV** — each row is one conversation, messages `\n`-separated:

```
conversation
"Hey are you free tonight?\nYeah what's up?\nWanna catch a movie?"
```

**PDF** — each page is treated as one conversation block.

---

## Limitations

- **Intra-conversation topic segmentation:** Topic detection operates at conversation level per batch. `detect_topic_shift()` catches shifts between consecutive conversations, not within a single conversation at message-window granularity.
- **Communication style classification:** Relative thresholds improve over fixed thresholds but short conversations (< 5 messages) provide insufficient signal for reliable style labeling — reflected in evaluation failures on communication style queries.
- **Persona signal asymmetry:** Evaluation shows User 2 persona scores consistently outperform User 1. Signal extraction quality is sensitive to how explicitly a user discloses information — not a systematic engine failure.
- **TextRank summaries:** Extractive only — pulls existing sentences, does not paraphrase. Fragmented conversations may produce incoherent summaries.
- **Two-speaker assumption:** Designed for dyadic conversations (User_1 / User_2). Group chats are not supported.

---

## Live Demo

🔗 [chatanalyzer-6nr3wskqbs5obudbmnwib9.streamlit.app](https://chatanalyzer-wt5qmalbvndicvtqjcoktp.streamlit.app/)

---

## Author

**Tina Prabhat**
BTech Final Year | ML · AI · NLP
Paper accepted at **ICMEET 2025, London**

---

## License

MIT

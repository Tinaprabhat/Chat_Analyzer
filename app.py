import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.kb_store import is_kb_populated
from src.query_engine import answer_query
from src.pipeline import run_pipeline
from src.config import CSV_PATH, PDF_PATH

st.set_page_config(
    page_title="Chat Persona RAG",
    page_icon="🧠",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    
    .main { background: #0d0d0d; color: #e8e8e8; }
    
    .header-block {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        border-left: 4px solid #e94560;
        margin-bottom: 2rem;
    }
    .header-block h1 {
        font-family: 'IBM Plex Mono', monospace;
        color: #e94560;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-block p { color: #a0a8b8; margin: 0.5rem 0 0 0; font-size: 0.95rem; }
    
    .answer-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 3px solid #10b981;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 0.97rem;
        line-height: 1.7;
        color: #d1fae5;
    }
    .context-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 3px solid #6366f1;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        max-height: 300px;
        overflow-y: auto;
    }
    .route-badge {
        display: inline-block;
        background: #1e293b;
        color: #818cf8;
        border: 1px solid #3730a3;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: #111827;
        border: 1px solid #1f2937;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .stat-card .num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        color: #e94560;
        font-weight: 600;
    }
    .stat-card .label { color: #6b7280; font-size: 0.82rem; margin-top: 2px; }
    
    .sample-btn {
        background: #1e293b;
        border: 1px solid #334155;
        color: #94a3b8;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-size: 0.82rem;
        cursor: pointer;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🧠 Chat Persona RAG</h1>
    <p>Conversation analysis · Persona extraction · Semantic retrieval · Grounded QA</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")

    kb_ready = is_kb_populated()
    if kb_ready:
        st.success("✅ Knowledge Base Ready")
    else:
        st.warning("⚠️ KB not built yet")

    st.markdown("---")
    st.markdown("### 🤖 LLM Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        key="groq",
        help="Primary LLM. Falls back to local Ollama if not set."
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    if os.getenv("GROQ_API_KEY", ""):
        st.success("✅ Groq ready (primary LLM)")
    else:
        st.warning("⚠️ No Groq key — will use Ollama fallback")

    # Ollama fallback status
    import urllib.request as _urllib
    import json as _json
    _ollama_base  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _ollama_model = os.getenv("OLLAMA_MODEL",    "qwen2.5:1.5b")
    try:
        with _urllib.urlopen(f"{_ollama_base}/api/tags", timeout=2) as _resp:
            _models = [m["name"] for m in _json.loads(_resp.read()).get("models", [])]
        if _ollama_model in _models:
            st.info(f"🔄 Ollama fallback: `{_ollama_model}` ready")
        else:
            st.warning(f"⚠️ Ollama running but `{_ollama_model}` not pulled")
    except Exception:
        st.caption("🔄 Ollama fallback: offline (optional)")

    st.markdown("---")
    st.markdown("### 📂 Data")
    
    # CSV uploader
    st.write("**Upload Conversations (CSV)**")
    uploaded_csv = st.file_uploader("Upload conversations CSV", type=["csv"], key="csv_uploader")
    if uploaded_csv:
        import shutil
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "wb") as f:
            f.write(uploaded_csv.read())
        st.success("✅ CSV uploaded")
    
    # PDF uploader
    st.write("**Upload Conversations (PDF)**")
    uploaded_pdf = st.file_uploader("Upload conversations PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf:
        import shutil
        os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
        with open(PDF_PATH, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("✅ PDF uploaded")

    force_rebuild = st.checkbox("Force rebuild KB", value=False)

    if st.button("🚀 Build Knowledge Base", use_container_width=True):
        if not os.path.exists(CSV_PATH) and not os.path.exists(PDF_PATH):
            st.error("No CSV or PDF found. Upload one first.")
        else:
            with st.spinner("Building KB... this may take a while for large datasets"):
                try:
                    run_pipeline(force_rebuild=force_rebuild)
                    st.success("✅ KB built successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

    st.markdown("---")
    st.markdown("### ℹ️ Route Legend")
    st.markdown("""
    - 🟣 `persona` → persona JSON
    - 🔵 `topic` → topic vectors
    - 🟢 `summary` → batch summaries
    - ⚪ `all` → all layers
    """)
    st.markdown("### 🔧 Pipeline Info")
    st.markdown(
        "**KB build**: fully offline  \n"
        "KeyBERT topics · TextRank summaries · spaCy NER personas  \n\n"
        "**Query answering**: Groq → Ollama fallback"
    )

# ── Main area ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

if kb_ready:
    import sqlite3
    from src.config import META_DB
    try:
        conn = sqlite3.connect(META_DB)
        batch_count = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        cp_count    = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        conn.close()
    except Exception:
        batch_count, cp_count = 0, 0

    with col1:
        st.markdown(f'<div class="stat-card"><div class="num">{batch_count}</div><div class="label">Batches Processed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="num">{batch_count * 10}</div><div class="label">Conversations</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="num">{cp_count}</div><div class="label">Checkpoints</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="num">2</div><div class="label">User Personas</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 💬 Ask the Knowledge Base")

# Sample questions
st.markdown("**Quick questions:**")
sample_qs = [
    "What kind of person is User 1?",
    "What are the habits of User 2?",
    "How does User 1 communicate?",
    "What topics are most discussed?",
    "Give me a summary of the conversations",
    "What does User 2 do for work?"
]
cols = st.columns(3)
for i, q in enumerate(sample_qs):
    if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
        st.session_state["prefill_query"] = q

query_val = st.session_state.get("prefill_query", "")
user_query = st.text_input(
    "Enter your question:",
    value=query_val,
    placeholder="What kind of person is User 1? What are their habits?",
    key="main_query"
)

if st.button("🔍 Search Knowledge Base", use_container_width=True, type="primary"):
    if not kb_ready:
        st.error("Please build the Knowledge Base first (sidebar → Build Knowledge Base)")
    elif not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving from KB and generating answer..."):
            try:
                result = answer_query(user_query)
                st.markdown(f'<div class="route-badge">🔀 Route: {result["route"]}</div>',
                            unsafe_allow_html=True)
                st.markdown("#### 📋 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                            unsafe_allow_html=True)

                with st.expander("📚 Retrieved Context (KB chunks used)"):
                    st.markdown(f'<div class="context-box">{result["context"]}</div>',
                                unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Query failed: {e}")

    # Clear prefill
    if "prefill_query" in st.session_state:
        del st.session_state["prefill_query"]

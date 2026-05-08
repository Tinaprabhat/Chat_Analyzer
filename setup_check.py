"""
System and environment setup checker.
Run this FIRST before anything else.
Usage: python setup_check.py
"""
import sys
import os
import subprocess
import importlib
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# ── Colors ─────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")


def check_python():
    print(f"\n{BOLD}[1] Python Environment{RESET}")
    v = sys.version_info
    info(f"Python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if v.major == 3 and v.minor >= 9:
        ok("Python version OK (3.9+)")
    else:
        fail(f"Python 3.9+ required. Got {v.major}.{v.minor}")
    return v.major == 3 and v.minor >= 9


def check_packages():
    print(f"\n{BOLD}[2] Required Packages{RESET}")
    required = {
        "yake":                 "yake",
        "sentence_transformers":"sentence-transformers",
        "chromadb":             "chromadb",
        "groq":                 "groq",
        "streamlit":            "streamlit",
        "pandas":               "pandas",
        "numpy":                "numpy",
        "sklearn":              "scikit-learn",
        "pydantic":             "pydantic",
        "tqdm":                 "tqdm",
        "colorama":             "colorama",
        "dotenv":               "python-dotenv",
        "requests":             "requests",
        "pypdf":                "pypdf",
    }
    all_ok = True
    for import_name, pip_name in required.items():
        try:
            importlib.import_module(import_name)
            ok(f"{pip_name}")
        except ImportError:
            fail(f"{pip_name} NOT installed → pip install {pip_name}")
            all_ok = False

    return all_ok


def check_ollama():
    print(f"\n{BOLD}[3] Ollama (Local SLM){RESET}")
    try:
        result = subprocess.run(["ollama", "--version"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            ok(f"Ollama installed: {result.stdout.strip()}")
        else:
            fail("Ollama not found. Install from https://ollama.com")
            return False
    except FileNotFoundError:
        fail("Ollama not found. Install from https://ollama.com")
        return False
    except Exception as e:
        fail(f"Ollama check failed: {e}")
        return False

    # Check if qwen2.5:1.5b is pulled
    try:
        result = subprocess.run(["ollama", "list"],
                                capture_output=True, text=True, timeout=10)
        if "qwen2.5:1.5b" in result.stdout:
            ok("qwen2.5:1.5b model available")
        else:
            warn("qwen2.5:1.5b not found → run: ollama pull qwen2.5:1.5b")
    except Exception:
        warn("Could not check Ollama models list")
    return True


def check_api_keys():
    print(f"\n{BOLD}[4] API Keys & Config{RESET}")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if groq_key and groq_key != "your_groq_api_key_here":
        ok(f"GROQ_API_KEY set ({groq_key[:8]}...)")
    else:
        warn("GROQ_API_KEY not set → Set in .env file or via: $env:GROQ_API_KEY='your_key'")

    # Check ollama config
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    info(f"Ollama (fallback): {ollama_model} at {ollama_url}")

    if not groq_key:
        fail("GROQ_API_KEY not set. Required for primary query answering.")
        return False
    return True


def check_data():
    print(f"\n{BOLD}[5] Data & Directories{RESET}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from src.config import CSV_PATH, KB_DIR, LOG_DIR, DATA_DIR
        for d in [DATA_DIR, KB_DIR, LOG_DIR]:
            os.makedirs(d, exist_ok=True)
            ok(f"Directory OK: {d}")

        if os.path.exists(CSV_PATH):
            import pandas as pd
            df = pd.read_csv(CSV_PATH, nrows=2)
            ok(f"CSV found: {CSV_PATH} ({len(open(CSV_PATH).readlines())} lines)")
        else:
            warn(f"CSV not found at {CSV_PATH}")
            info("Place conversations.csv in the data/ folder")
    except Exception as e:
        fail(f"Config/data check failed: {e}")
        return False
    return True


def check_embedding_model():
    print(f"\n{BOLD}[6] Embedding Model{RESET}")
    try:
        from sentence_transformers import SentenceTransformer
        info("Loading all-MiniLM-L6-v2 (first run downloads ~90MB)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode(["test"])
        ok(f"all-MiniLM-L6-v2 loaded. Output dim: {vec.shape[1]}")
        return True
    except Exception as e:
        fail(f"Embedding model failed: {e}")
        return False


def main():
    print(f"\n{BOLD}{'='*55}")
    print("  CHAT PERSONA RAG — SETUP CHECK")
    print(f"{'='*55}{RESET}")

    results = {
        "python":   check_python(),
        "packages": check_packages(),
        "ollama":   check_ollama(),
        "api_keys": check_api_keys(),
        "data":     check_data(),
        "embedder": check_embedding_model(),
    }

    print(f"\n{BOLD}{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}{RESET}")
    all_pass = True
    for check, passed in results.items():
        if passed:
            ok(check)
        else:
            fail(check)
            if check in ("python", "packages", "embedder"):
                all_pass = False

    print()
    if all_pass:
        print(f"{GREEN}{BOLD}✅ System ready. Run the app:{RESET}")
        print(f"   streamlit run app.py")
    else:
        print(f"{RED}{BOLD}❌ Fix the issues above before running.{RESET}")
    print()


if __name__ == "__main__":
    main()

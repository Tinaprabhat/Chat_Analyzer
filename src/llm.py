import os
import requests
from groq import Groq
from src.config import GROQ_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL
from src.logger import logger


def _call_groq(prompt: str) -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def call_llm(prompt: str) -> str:
    """Groq primary, Ollama fallback. Returns empty string if both fail."""
    if os.getenv("GROQ_API_KEY", ""):
        try:
            return _call_groq(prompt)
        except Exception as e:
            logger.warning(f"Groq failed: {e}. Falling back to Ollama...")
    try:
        return _call_ollama(prompt)
    except Exception as e:
        logger.error(f"Ollama fallback also failed: {e}")
        return ""

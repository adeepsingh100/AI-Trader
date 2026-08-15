from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_CHAIN = [
    m.strip()
    # `or` (not getenv's default arg) — GitHub Actions sets a referenced
    # secret that was never created as an empty string, not an absent
    # key, so getenv's own default never kicks in.
    for m in (os.getenv("GROQ_MODEL_CHAIN") or "openai/gpt-oss-120b,qwen/qwen3.6-27b").split(",")
    if m.strip()
]

# "groq" (default) or "ollama" (Ollama Cloud — https://ollama.com, not a
# local instance). Switch by setting LLM_PROVIDER, no code change needed.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "https://ollama.com"
OLLAMA_MODEL_CHAIN = [
    m.strip() for m in (os.getenv("OLLAMA_MODEL_CHAIN") or "gpt-oss:120b").split(",") if m.strip()
]

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_API_SECRET = os.getenv("COINDCX_API_SECRET", "")

PROMOTION_MIN_PAPER_DAYS = int(os.getenv("PROMOTION_MIN_PAPER_DAYS", "14"))
PROMOTION_MIN_CUMULATIVE_PNL = float(os.getenv("PROMOTION_MIN_CUMULATIVE_PNL", "0"))
PROMOTION_MAX_DRAWDOWN_PCT = float(os.getenv("PROMOTION_MAX_DRAWDOWN_PCT", "15"))

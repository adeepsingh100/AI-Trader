from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_CHAIN = [
    m.strip()
    for m in os.getenv("GROQ_MODEL_CHAIN", "openai/gpt-oss-120b,qwen/qwen3.6-27b").split(",")
    if m.strip()
]

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_API_SECRET = os.getenv("COINDCX_API_SECRET", "")

PROMOTION_MIN_PAPER_DAYS = int(os.getenv("PROMOTION_MIN_PAPER_DAYS", "14"))
PROMOTION_MIN_CUMULATIVE_PNL = float(os.getenv("PROMOTION_MIN_CUMULATIVE_PNL", "0"))
PROMOTION_MAX_DRAWDOWN_PCT = float(os.getenv("PROMOTION_MAX_DRAWDOWN_PCT", "15"))

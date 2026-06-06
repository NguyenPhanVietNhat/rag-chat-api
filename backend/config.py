"""
config.py  —  Cấu hình tập trung, đọc từ .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Claude AI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Auth / JWT ────────────────────────────────────────────────────────────────
SECRET_KEY:       str = os.getenv("SECRET_KEY", "change-me-in-production-32chars!")
ALGORITHM:        str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("TOKEN_EXPIRE_MIN", "1440"))  # 24h

# ── Storage ───────────────────────────────────────────────────────────────────
UPLOAD_DIR:   Path = Path(os.getenv("UPLOAD_DIR",  "./uploads"))
CHROMA_DIR:   str  = os.getenv("CHROMA_DIR",       "./chroma_db")
MAX_FILE_MB:  int  = int(os.getenv("MAX_FILE_MB",  "50"))

# ── RAG tuning ────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int   = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP: int   = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K:         int   = int(os.getenv("TOP_K",         "5"))
MIN_SCORE:     float = float(os.getenv("MIN_SCORE",   "0.20"))
EMBED_MODEL:   str   = os.getenv("EMBED_MODEL", "multilingual-default")

# ── Server ────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS: list[str] = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

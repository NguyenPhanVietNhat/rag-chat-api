# ════════════════════════════════════════════════════════════
#  Dockerfile — DocMind RAG Backend
#  Base: Python 3.11 slim (nhẹ, ổn định với AI libs)
# ════════════════════════════════════════════════════════════

# ── Stage 1: Build dependencies ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime image ───────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy code backend
COPY backend/ .

# Copy frontend để FastAPI serve tĩnh
COPY frontend/ ./frontend/

# Tạo thư mục cần thiết
RUN mkdir -p uploads chroma_db data

# Tải model embedding trước để không cần internet lúc chạy  ← THÊM ĐOẠN NÀY
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENAI_API_KEY="" \
    SECRET_KEY="change-me-in-production-32chars!" \
    UPLOAD_DIR="./uploads" \
    CHROMA_DIR="./chroma_db" \
    USER_DB_PATH="./data/users.db"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

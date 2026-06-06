"""
embedding_engine.py
───────────────────
Tạo vector embeddings từ văn bản dùng sentence-transformers.
Hỗ trợ batch, cache, nhiều model, và tích hợp trực tiếp với ChromaDB.

Dùng độc lập:
    engine = EmbeddingEngine()
    result = engine.embed(["câu 1", "câu 2"])
    print(result.vectors.shape)  # (2, 384)

Dùng với pipeline RAG:
    store = VectorStore()
    store.add_chunks(result.chunks)       # từ DocumentProcessor
    hits  = store.search("câu hỏi", top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    # Đa ngôn ngữ — khuyến nghị mặc định cho tiếng Việt
    "multilingual-default": {
        "name":       "paraphrase-multilingual-MiniLM-L12-v2",
        "dim":        384,
        "max_tokens": 128,
        "note":       "Nhanh, nhẹ, tốt cho tiếng Việt. Khuyến nghị mặc định.",
    },
    "multilingual-large": {
        "name":       "paraphrase-multilingual-mpnet-base-v2",
        "dim":        768,
        "max_tokens": 128,
        "note":       "Chất lượng cao hơn, nặng hơn.",
    },
    # Tiếng Anh
    "english-fast": {
        "name":       "all-MiniLM-L6-v2",
        "dim":        384,
        "max_tokens": 256,
        "note":       "Rất nhanh, phù hợp prototype tiếng Anh.",
    },
    "english-quality": {
        "name":       "all-mpnet-base-v2",
        "dim":        768,
        "max_tokens": 384,
        "note":       "Chất lượng tốt nhất cho tiếng Anh.",
    },
    # Passage retrieval
    "msmarco": {
        "name":       "msmarco-MiniLM-L6-cos-v5",
        "dim":        384,
        "max_tokens": 512,
        "note":       "Tối ưu cho query ngắn → đoạn văn dài.",
    },
}

DEFAULT_MODEL_KEY = "multilingual-default"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbedResult:
    """Kết quả embed một batch văn bản."""
    texts:      list[str]
    vectors:    np.ndarray        # shape (n, dim)
    model_name: str
    dim:        int
    elapsed_ms: float
    cache_hits: int = 0

    @property
    def count(self) -> int:
        return len(self.texts)

    def as_list(self) -> list[list[float]]:
        """Chuyển numpy → list[list[float]] để JSON-serialize / Chroma."""
        return self.vectors.tolist()

    def similarity(self, i: int, j: int) -> float:
        """Cosine similarity giữa vector i và j trong batch này."""
        a, b = self.vectors[i], self.vectors[j]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@dataclass
class SearchHit:
    """Một kết quả tìm kiếm semantic từ VectorStore."""
    text:        str
    score:       float              # cosine similarity [0, 1]
    chunk_id:    str   = ""
    doc_id:      str   = ""
    source_file: str   = ""
    page:        Optional[int] = None
    section:     Optional[str] = None
    metadata:    dict  = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    Cache in-memory: text → vector.
    Key = SHA-256(model_name + text) để tránh collision giữa các model.
    Tùy chọn persist ra disk (JSON).
    """

    def __init__(self, max_size: int = 10_000, cache_file: Optional[str] = None):
        self._store: dict[str, list[float]] = {}
        self._max  = max_size
        self._file = Path(cache_file) if cache_file else None
        self._hits = 0
        self._miss = 0
        if self._file and self._file.exists():
            self._load()

    def _key(self, model: str, text: str) -> str:
        return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> Optional[list[float]]:
        v = self._store.get(self._key(model, text))
        if v is not None:
            self._hits += 1
        else:
            self._miss += 1
        return v

    def set(self, model: str, text: str, vector: list[float]):
        if len(self._store) >= self._max:
            # Evict 20% cũ nhất (FIFO đơn giản)
            keys = list(self._store.keys())
            for k in keys[: len(keys) // 5]:
                del self._store[k]
        self._store[self._key(model, text)] = vector

    def save(self):
        if self._file:
            self._file.write_text(json.dumps(self._store))

    def _load(self):
        try:
            self._store = json.loads(self._file.read_text())
        except Exception:
            self._store = {}

    @property
    def stats(self) -> dict:
        total = self._hits + self._miss
        return {
            "size":     len(self._store),
            "hits":     self._hits,
            "misses":   self._miss,
            "hit_rate": round(self._hits / total, 3) if total else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Bộ tạo embeddings trung tâm dùng sentence-transformers.

    Tính năng:
      - Load model sentence-transformers (auto-download lần đầu)
      - Batch encoding với cache lookup
      - L2-normalize → dot-product = cosine (nhanh hơn khi query)
      - Hỗ trợ query prefix (asymmetric retrieval cho MSMARCO)
      - Đổi model runtime qua switch_model()

    Ví dụ:
        engine = EmbeddingEngine("multilingual-default", use_cache=True)
        result = engine.embed(["Chính sách bảo mật", "Điều khoản thanh toán"])
        print(result.vectors.shape)   # (2, 384)
        print(result.similarity(0, 1))
    """

    def __init__(
        self,
        model_key:  str  = DEFAULT_MODEL_KEY,
        batch_size: int  = 64,
        use_cache:  bool = True,
        cache_file: Optional[str] = None,
        normalize:  bool = True,
        device:     Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.normalize  = normalize
        self.cache      = EmbeddingCache(cache_file=cache_file) if use_cache else None

        self._model:      Optional[SentenceTransformer] = None
        self._model_name: str = ""
        self._dim:        int = 0

        self._load_model(model_key, device)

    # ── Model management ──────────────────────────────────────────────────────

    def _load_model(self, model_key: str, device: Optional[str] = None):
        info       = MODEL_REGISTRY.get(model_key, {"name": model_key, "dim": 0, "note": "custom"})
        model_name = info["name"]
        if model_name == self._model_name:
            return

        logger.info(f"Loading model: {model_name}")
        t0 = time.time()
        self._model      = SentenceTransformer(model_name, device=device or "cpu")
        self._model_name = model_name
        self._dim        = info.get("dim") or self._model.get_sentence_embedding_dimension()
        logger.info(f"  Loaded in {time.time()-t0:.1f}s | dim={self._dim}")

    def switch_model(self, model_key: str, device: Optional[str] = None):
        """Đổi model mà không cần tạo instance mới."""
        self._load_model(model_key, device)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    # ── Core embed ────────────────────────────────────────────────────────────

    def embed(
        self,
        texts:         list[str],
        is_query:      bool = False,
        show_progress: bool = False,
    ) -> EmbedResult:
        """
        Embed một list văn bản.

        Tham số:
            texts        – list chuỗi cần encode
            is_query     – True nếu đây là câu hỏi (dùng query prefix cho MSMARCO)
            show_progress – hiện tqdm progress bar

        Trả về EmbedResult (.vectors shape = (n, dim)).
        """
        if not texts:
            return EmbedResult([], np.empty((0, self._dim)), self._model_name, self._dim, 0)

        t0 = time.time()

        # Prefix asymmetric (chỉ áp dụng cho msmarco)
        use_prefix = "msmarco" in self._model_name
        prefix     = ("query: " if is_query else "passage: ") if use_prefix else ""
        prepared   = [prefix + t for t in texts]

        # ── Cache lookup ──────────────────────────────────────────────────────
        vectors   = [None] * len(prepared)
        to_encode: list[tuple[int, str]] = []
        cache_hits = 0

        if self.cache:
            for i, t in enumerate(prepared):
                cached = self.cache.get(self._model_name, t)
                if cached is not None:
                    vectors[i] = cached
                    cache_hits += 1
                else:
                    to_encode.append((i, t))
        else:
            to_encode = list(enumerate(prepared))

        # ── Encode texts chưa có trong cache ─────────────────────────────────
        if to_encode:
            indices, batch_texts = zip(*to_encode)
            raw: np.ndarray = self._model.encode(
                list(batch_texts),
                batch_size         = self.batch_size,
                normalize_embeddings = self.normalize,
                show_progress_bar  = show_progress,
                convert_to_numpy   = True,
            )
            for idx, vec, orig in zip(indices, raw, batch_texts):
                vec_list     = vec.tolist()
                vectors[idx] = vec_list
                if self.cache:
                    self.cache.set(self._model_name, orig, vec_list)

        arr     = np.array(vectors, dtype=np.float32)
        elapsed = (time.time() - t0) * 1000

        if len(texts) > 1:
            logger.info(
                f"Embedded {len(texts)} texts | {elapsed:.0f}ms | "
                f"cache={cache_hits}/{len(texts)}"
            )

        return EmbedResult(
            texts      = texts,
            vectors    = arr,
            model_name = self._model_name,
            dim        = self._dim,
            elapsed_ms = elapsed,
            cache_hits = cache_hits,
        )

    def embed_one(self, text: str, is_query: bool = False) -> np.ndarray:
        """Embed một chuỗi → numpy array (dim,)."""
        return self.embed([text], is_query=is_query).vectors[0]

    # ── Similarity ────────────────────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.normalize:
            return float(np.dot(a, b))
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (norm + 1e-9))

    def similarity_matrix(self, result: EmbedResult) -> np.ndarray:
        """Ma trận cosine similarity n×n."""
        V = result.vectors
        if self.normalize:
            return np.dot(V, V.T)
        norms  = np.linalg.norm(V, axis=1, keepdims=True)
        V_norm = V / (norms + 1e-9)
        return np.dot(V_norm, V_norm.T)

    def most_similar(
        self,
        query_vec:   np.ndarray,
        corpus_vecs: np.ndarray,
        top_k:       int = 5,
    ) -> list[tuple[int, float]]:
        """Top-k indices trong corpus_vecs gần nhất với query_vec."""
        if self.normalize:
            scores = corpus_vecs @ query_vec
        else:
            scores = corpus_vecs @ query_vec / (
                np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9
            )
        top = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top]

    @property
    def cache_stats(self) -> dict:
        return self.cache.stats if self.cache else {}


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE  —  ChromaDB wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Lưu trữ và tìm kiếm embeddings trong ChromaDB.

    - Upsert chunks từ DocumentProcessor
    - Tìm kiếm semantic với filter user_id / doc_id / page
    - Ghép context sẵn sàng đưa vào AI Agent
    - Xoá theo doc hoặc user

    Ví dụ:
        store = VectorStore(persist_dir="./chroma_db")
        store.add_chunks(result.chunks)
        hits  = store.search("điều khoản thanh toán", user_id="u_001")
        ctx   = store.build_context(hits)
    """

    def __init__(
        self,
        engine:      Optional[EmbeddingEngine] = None,
        persist_dir: Optional[str] = "./chroma_db",
        collection:  str = "rag_documents",
        distance:    str = "cosine",
    ):
        import chromadb

        self.engine = engine or EmbeddingEngine()

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._col = self._client.get_or_create_collection(
            name     = collection,
            metadata = {"hnsw:space": distance},
        )
        logger.info(
            f"VectorStore ready | collection='{collection}' | "
            f"existing={self._col.count()} chunks"
        )

    # ── Add ───────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list, batch_size: int = 100) -> int:
        """
        Nhận list[Chunk] từ DocumentProcessor và upsert vào Chroma.
        Tự động bỏ qua chunk đã tồn tại.
        Trả về số chunk đã thêm.
        """
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch  = chunks[i : i + batch_size]
            texts  = [c.text for c in batch]
            result = self.engine.embed(texts)

            self._col.upsert(
                ids        = [c.chunk_id for c in batch],
                embeddings = result.as_list(),
                documents  = texts,
                metadatas  = [
                    {
                        "doc_id":      c.doc_id,
                        "source_file": c.source_file,
                        "doc_type":    c.doc_type,
                        "user_id":     c.user_id,
                        "chunk_index": c.chunk_index,
                        "page":        c.page if c.page is not None else -1,
                        "section":     c.section or "",
                        "token_count": c.token_count,
                        "created_at":  c.created_at,
                    }
                    for c in batch
                ],
            )
            added += len(batch)

        logger.info(f"Added {added} chunks to VectorStore")
        return added

    def add_texts(
        self,
        texts:     list[str],
        metadatas: Optional[list[dict]] = None,
        ids:       Optional[list[str]]  = None,
    ) -> int:
        """Thêm text thô (không cần Chunk object)."""
        if not texts:
            return 0
        result = self.engine.embed(texts)
        _ids   = ids or [hashlib.sha256(t.encode()).hexdigest()[:12] for t in texts]
        _metas = metadatas or [{} for _ in texts]
        self._col.upsert(ids=_ids, embeddings=result.as_list(),
                         documents=texts, metadatas=_metas)
        return len(texts)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:     str,
        top_k:     int   = 5,
        user_id:   Optional[str] = None,
        doc_id:    Optional[str] = None,
        page:      Optional[int] = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """
        Tìm kiếm semantic: câu hỏi → top-k chunks phù hợp nhất.

        Tham số:
            query     – câu hỏi người dùng
            top_k     – số kết quả tối đa
            user_id   – lọc theo user (bắt buộc trong production)
            doc_id    – lọc trong một tài liệu cụ thể
            page      – lọc theo trang (PDF)
            min_score – loại bỏ kết quả có score thấp hơn ngưỡng
        """
        if self._col.count() == 0:
            return []

        # Build Chroma where clause
        conditions = []
        if user_id:
            conditions.append({"user_id": {"$eq": user_id}})
        if doc_id:
            conditions.append({"doc_id": {"$eq": doc_id}})
        if page is not None:
            conditions.append({"page": {"$eq": page}})

        where: dict = {}
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        query_vec = self.engine.embed_one(query, is_query=True)
        # embed_one có thể trả về numpy array hoặc list — xử lý cả hai
        _qvec_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)

        kwargs: dict = dict(
            query_embeddings = [_qvec_list],
            n_results        = min(top_k * 2, max(1, self._col.count())),
            include          = ["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw       = self._col.query(**kwargs)
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        hits: list[SearchHit] = []
        for text, meta, dist in zip(docs, metas, distances):
            score = round(1.0 - dist, 4)   # cosine distance → similarity
            if score < min_score:
                continue
            hits.append(SearchHit(
                text        = text,
                score       = score,
                doc_id      = meta.get("doc_id", ""),
                source_file = meta.get("source_file", ""),
                page        = meta.get("page") if meta.get("page", -1) >= 0 else None,
                section     = meta.get("section") or None,
                metadata    = meta,
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def build_context(
        self,
        hits:        list[SearchHit],
        max_tokens:  int  = 2000,
        cite_source: bool = True,
    ) -> str:
        """
        Ghép các SearchHit thành chuỗi context để đưa vào AI Agent.
        Tự cắt nếu ước tính vượt max_tokens.
        """
        parts  = []
        tokens = 0
        for i, hit in enumerate(hits, 1):
            loc    = f"trang {hit.page}" if hit.page else (hit.section or "")
            header = f"[Nguồn {i}: {hit.source_file}" + (f", {loc}" if loc else "") + "]\n" \
                     if cite_source else ""
            block  = header + hit.text
            approx = len(block) // 3
            if tokens + approx > max_tokens:
                break
            parts.append(block)
            tokens += approx
        return "\n\n---\n\n".join(parts)

    # ── Management ────────────────────────────────────────────────────────────

    def delete_doc(self, doc_id: str) -> int:
        existing = self._col.get(where={"doc_id": {"$eq": doc_id}})
        ids = existing["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def delete_user(self, user_id: str) -> int:
        existing = self._col.get(where={"user_id": {"$eq": user_id}})
        ids = existing["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    @property
    def count(self) -> int:
        return self._col.count()

    def stats(self) -> dict:
        return {
            "total_chunks": self.count,
            "model":        self.engine.model_name,
            "embed_dim":    self.engine.dim,
            "cache":        self.engine.cache_stats,
        }

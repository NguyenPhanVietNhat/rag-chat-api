"""
vector_store.py
───────────────
Unified Vector Store interface hỗ trợ cả ChromaDB và Qdrant.
Đổi backend chỉ bằng 1 tham số, code dùng giống nhau hoàn toàn.

Cách dùng:
    # ChromaDB (local, dev)
    store = VectorStore.chroma(persist_dir="./chroma_db")

    # Qdrant (local in-memory, test)
    store = VectorStore.qdrant_memory()

    # Qdrant (self-hosted server)
    store = VectorStore.qdrant_server(host="localhost", port=6333)

    # Qdrant Cloud
    store = VectorStore.qdrant_cloud(url="https://xxx.qdrant.io", api_key="...")

    # API giống nhau cho tất cả backend:
    store.add(texts, embeddings, metadatas, ids)
    hits = store.search(query_vec, top_k=5, filters={"user_id": "u1"})
    store.delete_by_filter({"doc_id": "abc"})
    store.count()
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """Một kết quả tìm kiếm từ vector store."""
    id:       str
    text:     str
    score:    float
    metadata: dict = field(default_factory=dict)

    # Shortcuts cho RAG
    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "")

    @property
    def page(self) -> Optional[int]:
        p = self.metadata.get("page", -1)
        return p if p >= 0 else None

    @property
    def section(self) -> Optional[str]:
        return self.metadata.get("section") or None

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "")

    def __repr__(self) -> str:
        return f"<Hit score={self.score:.3f} src='{self.source_file}' page={self.page}>"


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseVectorStore(ABC):
    """Interface chung cho mọi backend."""

    @abstractmethod
    def add(
        self,
        texts:      list[str],
        embeddings: list[list[float]],
        metadatas:  list[dict],
        ids:        Optional[list[str]] = None,
    ) -> int:
        """Thêm vectors. Trả về số lượng đã thêm."""

    @abstractmethod
    def search(
        self,
        query_vec: list[float] | np.ndarray,
        top_k:     int = 5,
        filters:   Optional[dict] = None,
        min_score: float = -1.0,
    ) -> list[SearchResult]:
        """Tìm kiếm top-k vectors gần nhất."""

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> int:
        """Xoá theo list ID."""

    @abstractmethod
    def delete_by_filter(self, filters: dict) -> int:
        """Xoá tất cả vectors khớp filter."""

    @abstractmethod
    def count(self, filters: Optional[dict] = None) -> int:
        """Đếm số vector (tùy chọn có filter)."""

    @abstractmethod
    def get_by_filter(self, filters: dict) -> list[dict]:
        """Lấy metadata của các vectors khớp filter."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Tên backend: 'chroma' hoặc 'qdrant'."""

    def info(self) -> dict:
        return {"backend": self.backend_name, "count": self.count()}


# ═══════════════════════════════════════════════════════════════════════════════
# CHROMADB BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB backend.
    - Không cần server riêng — lưu thẳng vào disk (PersistentClient)
    - Phù hợp: dev local, dự án nhỏ, ≤ 1 triệu vectors
    - HNSW index tích hợp sẵn

    Cài: pip install chromadb
    """

    def __init__(
        self,
        persist_dir: Optional[str] = "./chroma_db",
        collection:  str           = "rag_documents",
        distance:    str           = "cosine",      # cosine | l2 | ip
    ):
        import chromadb

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB PersistentClient @ {persist_dir}")
        else:
            self._client = chromadb.Client()
            logger.info("ChromaDB EphemeralClient (in-memory)")

        self._col = self._client.get_or_create_collection(
            name     = collection,
            metadata = {"hnsw:space": distance},
        )
        logger.info(f"Collection '{collection}' ready — {self._col.count()} vectors")

    @property
    def backend_name(self) -> str:
        return "chroma"

    def add(self, texts, embeddings, metadatas, ids=None) -> int:
        if not texts:
            return 0
        _ids = ids or [str(uuid.uuid4()) for _ in texts]

        # ChromaDB chỉ chấp nhận scalar trong metadata
        clean_metas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = str(v)
            clean_metas.append(clean)

        self._col.upsert(
            ids        = _ids,
            embeddings = [e if isinstance(e, list) else e.tolist() for e in embeddings],
            documents  = texts,
            metadatas  = clean_metas,
        )
        return len(texts)

    def search(self, query_vec, top_k=5, filters=None, min_score=-1.0) -> list[SearchResult]:
        total = self._col.count()
        if total == 0:
            return []

        vec = query_vec.tolist() if isinstance(query_vec, np.ndarray) else query_vec

        # Build Chroma where clause từ flat dict filters
        where: dict = {}
        if filters:
            conditions = [
                {k: {"$eq": v}} for k, v in filters.items()
                if v is not None
            ]
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}

        # n_results cannot exceed filtered subset — pre-count to avoid Chroma error
        if where:
            filtered_count = len(self._col.get(where=where)["ids"])
            n_req = min(top_k, filtered_count) if filtered_count > 0 else 0
        else:
            n_req = min(top_k, total)

        if n_req == 0:
            return []

        kwargs: dict = dict(
            query_embeddings = [vec],
            n_results        = n_req,
            include          = ["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw       = self._col.query(**kwargs)
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]
        raw_ids   = raw["ids"][0]

        results = []
        for rid, text, meta, dist in zip(raw_ids, docs, metas, distances):
            # Chroma cosine distance ∈ [0, 2]; clamp to [0, 1]
            score = round(max(0.0, 1.0 - dist), 4)
            if score < min_score:
                continue
            results.append(SearchResult(id=rid, text=text, score=score, metadata=meta))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        self._col.delete(ids=ids)
        return len(ids)

    def delete_by_filter(self, filters: dict) -> int:
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        existing = self._col.get(where=where)
        ids = existing["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def count(self, filters=None) -> int:
        if filters:
            conditions = [{k: {"$eq": v}} for k, v in filters.items()]
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
            return len(self._col.get(where=where)["ids"])
        return self._col.count()

    def get_by_filter(self, filters: dict) -> list[dict]:
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        raw = self._col.get(where=where, include=["metadatas", "documents"])
        return [
            {"id": rid, "text": text, **meta}
            for rid, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# QDRANT BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant backend — hỗ trợ 3 chế độ:
      1. In-memory  (test, không persist)
      2. Local disk (file-based, persist)
      3. Server     (self-hosted hoặc Qdrant Cloud)

    Ưu điểm Qdrant so với Chroma:
      - Filter phức tạp hơn (range, geo, nested)
      - Payload indexing → filter nhanh hơn với dữ liệu lớn
      - Hỗ trợ nhiều vector cùng lúc (sparse + dense)
      - REST + gRPC API khi dùng server
      - Scale tốt hơn (hàng triệu vectors)

    Cài: pip install qdrant-client
    Docker: docker run -p 6333:6333 qdrant/qdrant
    """

    def __init__(
        self,
        client,                          # QdrantClient instance
        collection:  str  = "rag_documents",
        vector_dim:  int  = 384,
        distance:    str  = "Cosine",    # Cosine | Euclid | Dot
        on_disk:     bool = False,       # lưu vectors ra disk (tiết kiệm RAM)
    ):
        from qdrant_client.models import Distance, VectorParams

        self._client     = client
        self._collection = collection
        self._dim        = vector_dim

        dist_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot":    Distance.DOT,
        }

        # Tạo collection nếu chưa có
        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            self._client.create_collection(
                collection_name = collection,
                vectors_config  = VectorParams(
                    size     = vector_dim,
                    distance = dist_map.get(distance, Distance.COSINE),
                    on_disk  = on_disk,
                ),
            )
            # Index payload fields để filter nhanh
            self._create_payload_indexes()
            logger.info(f"Qdrant collection '{collection}' created (dim={vector_dim})")
        else:
            logger.info(f"Qdrant collection '{collection}' loaded — {self.count()} vectors")

    def _create_payload_indexes(self):
        """Tạo index cho các field hay dùng trong filter."""
        from qdrant_client.models import PayloadSchemaType
        for field_name in ["user_id", "doc_id", "source_file"]:
            try:
                self._client.create_payload_index(
                    collection_name = self._collection,
                    field_name      = field_name,
                    field_schema    = PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass   # index đã tồn tại

    @property
    def backend_name(self) -> str:
        return "qdrant"

    def add(self, texts, embeddings, metadatas, ids=None) -> int:
        from qdrant_client.models import PointStruct

        if not texts:
            return 0
        _ids = ids or [str(uuid.uuid4()) for _ in texts]

        # Qdrant point ID phải là integer hoặc UUID string
        def to_point_id(s: str) -> str:
            # Đảm bảo là valid UUID (Qdrant chấp nhận UUID string)
            try:
                uuid.UUID(s)
                return s
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

        points = [
            PointStruct(
                id      = to_point_id(_ids[i]),
                vector  = embeddings[i] if isinstance(embeddings[i], list)
                          else embeddings[i].tolist(),
                payload = {"text": texts[i], **metadatas[i]},
            )
            for i in range(len(texts))
        ]

        self._client.upsert(
            collection_name = self._collection,
            points          = points,
            wait            = True,
        )
        return len(points)

    def search(self, query_vec, top_k=5, filters=None, min_score=-1.0) -> list[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        vec = query_vec.tolist() if isinstance(query_vec, np.ndarray) else query_vec

        # Build Qdrant Filter từ flat dict
        qdrant_filter = None
        if filters:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
                if v is not None
            ]
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        raw = self._client.query_points(
            collection_name = self._collection,
            query           = vec,
            limit           = top_k * 2,
            query_filter    = qdrant_filter,
            with_payload    = True,
        ).points

        results = []
        for hit in raw:
            payload = dict(hit.payload or {})
            text    = payload.pop("text", "")
            score   = round(float(hit.score), 4)
            if score < min_score:
                continue
            results.append(SearchResult(
                id       = str(hit.id),
                text     = text,
                score    = score,
                metadata = payload,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name = self._collection,
            points_selector = PointIdsList(points=ids),
            wait            = True,
        )
        return len(ids)

    def delete_by_filter(self, filters: dict) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        before = self.count(filters)
        must   = [FieldCondition(key=k, match=MatchValue(value=v))
                  for k, v in filters.items()]
        self._client.delete(
            collection_name = self._collection,
            points_selector = FilterSelector(filter=Filter(must=must)),
            wait            = True,
        )
        return before

    def count(self, filters=None) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if filters:
            must = [FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()]
            result = self._client.count(
                collection_name = self._collection,
                count_filter    = Filter(must=must),
                exact           = True,
            )
        else:
            result = self._client.count(
                collection_name = self._collection,
                exact           = True,
            )
        return result.count

    def get_by_filter(self, filters: dict) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must = [FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()]
        records, _ = self._client.scroll(
            collection_name = self._collection,
            scroll_filter   = Filter(must=must),
            with_payload    = True,
            limit           = 1000,
        )
        results = []
        for r in records:
            payload = r.payload or {}
            results.append({"id": str(r.id), **payload})
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY  —  dùng VectorStore.chroma() / VectorStore.qdrant_*()
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Factory class: tạo backend phù hợp.

    Ví dụ:
        store = VectorStore.chroma()                           # dev local
        store = VectorStore.qdrant_memory()                    # test
        store = VectorStore.qdrant_local("./qdrant_data")      # persist local
        store = VectorStore.qdrant_server("localhost", 6333)   # self-hosted
        store = VectorStore.qdrant_cloud(url, api_key)         # cloud
    """

    @staticmethod
    def chroma(
        persist_dir: Optional[str] = "./chroma_db",
        collection:  str           = "rag_documents",
        distance:    str           = "cosine",
    ) -> ChromaVectorStore:
        return ChromaVectorStore(persist_dir, collection, distance)

    @staticmethod
    def qdrant_memory(
        collection: str = "rag_documents",
        vector_dim: int = 384,
        distance:   str = "Cosine",
    ) -> QdrantVectorStore:
        """In-memory — dùng để test, không persist."""
        from qdrant_client import QdrantClient
        return QdrantVectorStore(
            QdrantClient(":memory:"), collection, vector_dim, distance
        )

    @staticmethod
    def qdrant_local(
        path:       str = "./qdrant_data",
        collection: str = "rag_documents",
        vector_dim: int = 384,
        distance:   str = "Cosine",
    ) -> QdrantVectorStore:
        """Local disk persist — không cần Docker."""
        from qdrant_client import QdrantClient
        return QdrantVectorStore(
            QdrantClient(path=path), collection, vector_dim, distance
        )

    @staticmethod
    def qdrant_server(
        host:       str = "localhost",
        port:       int = 6333,
        collection: str = "rag_documents",
        vector_dim: int = 384,
        distance:   str = "Cosine",
        api_key:    Optional[str] = None,
    ) -> QdrantVectorStore:
        """Self-hosted Qdrant server (Docker)."""
        from qdrant_client import QdrantClient
        return QdrantVectorStore(
            QdrantClient(host=host, port=port, api_key=api_key),
            collection, vector_dim, distance,
        )

    @staticmethod
    def qdrant_cloud(
        url:        str,
        api_key:    str,
        collection: str = "rag_documents",
        vector_dim: int = 384,
        distance:   str = "Cosine",
    ) -> QdrantVectorStore:
        """Qdrant Cloud — https://cloud.qdrant.io"""
        from qdrant_client import QdrantClient
        return QdrantVectorStore(
            QdrantClient(url=url, api_key=api_key),
            collection, vector_dim, distance,
        )

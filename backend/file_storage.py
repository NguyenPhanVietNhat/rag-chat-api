"""
file_storage.py
───────────────
Lưu trữ tài liệu gốc (PDF, DOCX, XLSX...) với interface thống nhất.
Hỗ trợ: MinIO (self-hosted) · AWS S3 · Cloudflare R2 · Local disk (dev)

Đổi backend chỉ bằng 1 dòng — code upload/download giống nhau hoàn toàn.

Cách dùng:
    # Dev local (không cần server)
    storage = FileStorage.local("./uploads")

    # MinIO (self-hosted, Docker)
    storage = FileStorage.minio("localhost:9000", "minioadmin", "minioadmin")

    # AWS S3
    storage = FileStorage.s3("my-bucket", region="ap-southeast-1")

    # Cloudflare R2 (~$0/tháng với 10GB)
    storage = FileStorage.r2(account_id, access_key, secret_key)

    # API giống nhau cho tất cả:
    url  = storage.upload(file_bytes, "user123/doc.pdf", content_type="application/pdf")
    data = storage.download("user123/doc.pdf")
    storage.delete("user123/doc.pdf")
    files = storage.list_files(prefix="user123/")
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileInfo:
    """Metadata của một file trong storage."""
    key:          str            # đường dẫn trong bucket: "user123/doc.pdf"
    size:         int            # bytes
    content_type: str
    etag:         str            # MD5 hash
    last_modified: Optional[str] = None
    url:          Optional[str]  = None  # presigned URL (nếu có)


@dataclass
class UploadResult:
    """Kết quả sau khi upload thành công."""
    key:          str     # "user123/hop_dong.pdf"
    bucket:       str
    size:         int
    content_type: str
    etag:         str
    backend:      str     # "minio" | "s3" | "r2" | "local"


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseFileStorage(ABC):
    """Interface chung cho mọi backend lưu file."""

    @abstractmethod
    def upload(
        self,
        data:         bytes | io.IOBase,
        key:          str,
        content_type: str = "application/octet-stream",
        metadata:     Optional[dict] = None,
    ) -> UploadResult:
        """Upload file. key = đường dẫn trong bucket (vd: 'u001/bao_cao.pdf')."""

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Download file → bytes."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Xoá file. Trả về True nếu thành công."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Kiểm tra file tồn tại."""

    @abstractmethod
    def get_info(self, key: str) -> FileInfo:
        """Lấy metadata của file."""

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[FileInfo]:
        """Liệt kê files theo prefix (vd: prefix='u001/' để xem file của user)."""

    @abstractmethod
    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        """Tạo URL tạm thời để download trực tiếp (bypass server)."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Tên backend."""

    # ── Helpers dùng chung ────────────────────────────────────────────────────

    @staticmethod
    def make_key(user_id: str, filename: str) -> str:
        """
        Tạo key an toàn từ user_id + filename.
        Format: {user_id}/{safe_filename}
        """
        # Chỉ giữ ký tự an toàn trong filename
        import re
        safe = re.sub(r"[^\w\-.]", "_", filename)
        safe = safe[:200]   # giới hạn độ dài
        return f"{user_id}/{safe}"

    @staticmethod
    def guess_content_type(filename: str) -> str:
        """Đoán Content-Type từ extension."""
        ext = Path(filename).suffix.lower()
        return {
            ".pdf":  "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc":  "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls":  "application/vnd.ms-excel",
            ".txt":  "text/plain; charset=utf-8",
            ".md":   "text/markdown; charset=utf-8",
            ".csv":  "text/csv; charset=utf-8",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(ext, "application/octet-stream")

    def info(self) -> dict:
        return {"backend": self.backend_name}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL DISK  (dev, không cần server)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalFileStorage(BaseFileStorage):
    """
    Lưu file trên disk local — dùng để dev, không cần server nào.
    Cấu trúc thư mục: {base_dir}/{key}

    Presigned URL trả về đường dẫn file local (không phải HTTP URL thật).
    """

    def __init__(self, base_dir: str = "./uploads"):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalFileStorage @ {self._base.resolve()}")

    @property
    def backend_name(self) -> str:
        return "local"

    def _path(self, key: str) -> Path:
        p = (self._base / key).resolve()
        # Security: đảm bảo không thoát ra ngoài base_dir
        if not str(p).startswith(str(self._base.resolve())):
            raise ValueError(f"Invalid key (path traversal): {key}")
        return p

    def upload(self, data, key, content_type="application/octet-stream", metadata=None) -> UploadResult:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw = data if isinstance(data, bytes) else data.read()
        path.write_bytes(raw)

        etag = hashlib.md5(raw).hexdigest()
        logger.info(f"[local] upload {key} ({len(raw)} bytes)")
        return UploadResult(
            key=key, bucket=str(self._base), size=len(raw),
            content_type=content_type, etag=etag, backend=self.backend_name,
        )

    def download(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.info(f"[local] deleted {key}")
            return True
        return False

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def get_info(self, key: str) -> FileInfo:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(key)
        stat = path.stat()
        raw  = path.read_bytes()
        return FileInfo(
            key=key, size=stat.st_size,
            content_type=self.guess_content_type(key),
            etag=hashlib.md5(raw).hexdigest(),
            last_modified=str(stat.st_mtime),
        )

    def list_files(self, prefix: str = "") -> list[FileInfo]:
        search_dir = self._base / prefix if prefix else self._base
        if not search_dir.exists():
            return []
        files = []
        for p in search_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self._base))
                try:
                    files.append(self.get_info(rel))
                except Exception:
                    pass
        return files

    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        # Local: trả về đường dẫn tuyệt đối (dùng khi serve static)
        return f"file://{self._path(key)}"


# ═══════════════════════════════════════════════════════════════════════════════
# MINIO  (self-hosted S3-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class MinIOFileStorage(BaseFileStorage):
    """
    MinIO — self-hosted, S3-compatible, miễn phí.
    Chạy local bằng Docker:
        docker run -d -p 9000:9000 -p 9001:9001 \\
            -e MINIO_ROOT_USER=minioadmin \\
            -e MINIO_ROOT_PASSWORD=minioadmin \\
            -v ./minio_data:/data \\
            minio/minio server /data --console-address ':9001'
        # Console: http://localhost:9001

    Cài: pip install minio
    """

    def __init__(
        self,
        endpoint:   str  = "localhost:9000",
        access_key: str  = "minioadmin",
        secret_key: str  = "minioadmin",
        bucket:     str  = "docmind",
        secure:     bool = False,           # True khi dùng HTTPS
        region:     str  = "us-east-1",
    ):
        from minio import Minio

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket = bucket
        self._region = region
        self._ensure_bucket()
        logger.info(f"MinIO @ {endpoint} bucket='{bucket}'")

    def _ensure_bucket(self):
        """Tạo bucket nếu chưa tồn tại."""
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket, location=self._region)
            logger.info(f"Created bucket '{self._bucket}'")

    @property
    def backend_name(self) -> str:
        return "minio"

    def upload(self, data, key, content_type="application/octet-stream", metadata=None) -> UploadResult:
        from minio.commonconfig import Tags

        raw    = data if isinstance(data, bytes) else data.read()
        stream = io.BytesIO(raw)
        size   = len(raw)

        result = self._client.put_object(
            bucket_name  = self._bucket,
            object_name  = key,
            data         = stream,
            length       = size,
            content_type = content_type,
            metadata     = metadata or {},
        )
        logger.info(f"[minio] upload {key} ({size} bytes) etag={result.etag}")
        return UploadResult(
            key=key, bucket=self._bucket, size=size,
            content_type=content_type, etag=result.etag or "",
            backend=self.backend_name,
        )

    def download(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, key: str) -> bool:
        try:
            self._client.remove_object(self._bucket, key)
            logger.info(f"[minio] deleted {key}")
            return True
        except Exception as e:
            logger.warning(f"[minio] delete failed {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except Exception:
            return False

    def get_info(self, key: str) -> FileInfo:
        stat = self._client.stat_object(self._bucket, key)
        return FileInfo(
            key=key,
            size=stat.size,
            content_type=stat.content_type or "",
            etag=(stat.etag or "").strip('"'),
            last_modified=str(stat.last_modified),
        )

    def list_files(self, prefix: str = "") -> list[FileInfo]:
        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        files = []
        for obj in objects:
            files.append(FileInfo(
                key=obj.object_name,
                size=obj.size or 0,
                content_type="",
                etag=(obj.etag or "").strip('"'),
                last_modified=str(obj.last_modified),
            ))
        return files

    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        url = self._client.presigned_get_object(
            self._bucket, key,
            expires=timedelta(seconds=expires_seconds),
        )
        return url


# ═══════════════════════════════════════════════════════════════════════════════
# AWS S3  /  CLOUDFLARE R2  (boto3-based)
# ═══════════════════════════════════════════════════════════════════════════════

class S3FileStorage(BaseFileStorage):
    """
    AWS S3 hoặc Cloudflare R2 (S3-compatible) qua boto3.

    AWS S3:
        storage = S3FileStorage(bucket="my-bucket", region="ap-southeast-1")
        # Credentials từ env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    Cloudflare R2 (~$0/tháng với 10GB free):
        storage = S3FileStorage(
            bucket        = "docmind",
            endpoint_url  = "https://<account_id>.r2.cloudflarestorage.com",
            access_key_id = "...",
            secret_key    = "...",
            region        = "auto",
        )

    Cài: pip install boto3
    """

    def __init__(
        self,
        bucket:          str,
        region:          str            = "ap-southeast-1",
        endpoint_url:    Optional[str]  = None,    # None = AWS S3 thật
        access_key_id:   Optional[str]  = None,    # None = đọc từ env/~/.aws
        secret_key:      Optional[str]  = None,
        public_base_url: Optional[str]  = None,    # CDN URL nếu có
    ):
        import boto3

        session = boto3.Session(
            aws_access_key_id     = access_key_id,
            aws_secret_access_key = secret_key,
            region_name           = region,
        )
        self._s3 = session.client(
            "s3",
            endpoint_url = endpoint_url,
        )
        self._bucket          = bucket
        self._region          = region
        self._endpoint_url    = endpoint_url
        self._public_base_url = public_base_url
        self._ensure_bucket()

        backend = "r2" if endpoint_url and "r2.cloudflarestorage" in endpoint_url else "s3"
        self._backend = backend
        logger.info(f"[{backend}] bucket='{bucket}' region='{region}'")

    def _ensure_bucket(self):
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                if self._region == "us-east-1" or self._endpoint_url:
                    self._s3.create_bucket(Bucket=self._bucket)
                else:
                    self._s3.create_bucket(
                        Bucket=self._bucket,
                        CreateBucketConfiguration={"LocationConstraint": self._region},
                    )
                logger.info(f"Created bucket '{self._bucket}'")
            except Exception as e:
                logger.warning(f"Bucket may already exist: {e}")

    @property
    def backend_name(self) -> str:
        return self._backend

    def upload(self, data, key, content_type="application/octet-stream", metadata=None) -> UploadResult:
        raw    = data if isinstance(data, bytes) else data.read()
        stream = io.BytesIO(raw)

        extra = {"ContentType": content_type}
        if metadata:
            extra["Metadata"] = {k: str(v) for k, v in metadata.items()}

        self._s3.upload_fileobj(stream, self._bucket, key, ExtraArgs=extra)

        # Lấy ETag
        head = self._s3.head_object(Bucket=self._bucket, Key=key)
        etag = head.get("ETag", "").strip('"')

        logger.info(f"[{self._backend}] upload {key} ({len(raw)} bytes)")
        return UploadResult(
            key=key, bucket=self._bucket, size=len(raw),
            content_type=content_type, etag=etag, backend=self.backend_name,
        )

    def download(self, key: str) -> bytes:
        buf = io.BytesIO()
        self._s3.download_fileobj(self._bucket, key, buf)
        return buf.getvalue()

    def delete(self, key: str) -> bool:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
            logger.info(f"[{self._backend}] deleted {key}")
            return True
        except Exception as e:
            logger.warning(f"[{self._backend}] delete failed: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def get_info(self, key: str) -> FileInfo:
        head = self._s3.head_object(Bucket=self._bucket, Key=key)
        return FileInfo(
            key=key,
            size=head["ContentLength"],
            content_type=head.get("ContentType", ""),
            etag=head.get("ETag", "").strip('"'),
            last_modified=str(head.get("LastModified", "")),
        )

    def list_files(self, prefix: str = "") -> list[FileInfo]:
        paginator = self._s3.get_paginator("list_objects_v2")
        files     = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                files.append(FileInfo(
                    key=obj["Key"],
                    size=obj["Size"],
                    content_type="",
                    etag=obj.get("ETag", "").strip('"'),
                    last_modified=str(obj.get("LastModified", "")),
                ))
        return files

    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        if self._public_base_url:
            return f"{self._public_base_url.rstrip('/')}/{key}"
        return self._s3.generate_presigned_url(
            "get_object",
            Params    = {"Bucket": self._bucket, "Key": key},
            ExpiresIn = expires_seconds,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

class FileStorage:
    """
    Factory: tạo backend phù hợp theo môi trường.

        storage = FileStorage.local()           # dev, không cần server
        storage = FileStorage.minio(...)        # self-hosted Docker
        storage = FileStorage.s3(...)           # AWS S3
        storage = FileStorage.r2(...)           # Cloudflare R2 (free tier)
        storage = FileStorage.from_env()        # đọc STORAGE_BACKEND từ .env
    """

    @staticmethod
    def local(base_dir: str = "./uploads") -> LocalFileStorage:
        return LocalFileStorage(base_dir)

    @staticmethod
    def minio(
        endpoint:   str  = "localhost:9000",
        access_key: str  = "minioadmin",
        secret_key: str  = "minioadmin",
        bucket:     str  = "docmind",
        secure:     bool = False,
    ) -> MinIOFileStorage:
        return MinIOFileStorage(endpoint, access_key, secret_key, bucket, secure)

    @staticmethod
    def s3(
        bucket:        str,
        region:        str           = "ap-southeast-1",
        access_key_id: Optional[str] = None,
        secret_key:    Optional[str] = None,
    ) -> S3FileStorage:
        return S3FileStorage(bucket, region,
                             access_key_id=access_key_id,
                             secret_key=secret_key)

    @staticmethod
    def r2(
        account_id: str,
        access_key: str,
        secret_key: str,
        bucket:     str = "docmind",
    ) -> S3FileStorage:
        """Cloudflare R2 — free tier: 10GB storage, 1M requests/tháng."""
        return S3FileStorage(
            bucket       = bucket,
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com",
            access_key_id = access_key,
            secret_key   = secret_key,
            region       = "auto",
        )

    @staticmethod
    def from_env() -> BaseFileStorage:
        """
        Đọc cấu hình từ biến môi trường.

        STORAGE_BACKEND=local|minio|s3|r2

        Local:
            STORAGE_BACKEND=local
            UPLOAD_DIR=./uploads

        MinIO:
            STORAGE_BACKEND=minio
            MINIO_ENDPOINT=localhost:9000
            MINIO_ACCESS_KEY=minioadmin
            MINIO_SECRET_KEY=minioadmin
            MINIO_BUCKET=docmind

        S3:
            STORAGE_BACKEND=s3
            AWS_ACCESS_KEY_ID=...
            AWS_SECRET_ACCESS_KEY=...
            S3_BUCKET=docmind
            AWS_DEFAULT_REGION=ap-southeast-1

        R2:
            STORAGE_BACKEND=r2
            R2_ACCOUNT_ID=...
            R2_ACCESS_KEY=...
            R2_SECRET_KEY=...
            R2_BUCKET=docmind
        """
        backend = os.getenv("STORAGE_BACKEND", "local").lower()

        if backend == "local":
            return FileStorage.local(os.getenv("UPLOAD_DIR", "./uploads"))

        if backend == "minio":
            return FileStorage.minio(
                endpoint   = os.getenv("MINIO_ENDPOINT",   "localhost:9000"),
                access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                bucket     = os.getenv("MINIO_BUCKET",     "docmind"),
                secure     = os.getenv("MINIO_SECURE", "false").lower() == "true",
            )

        if backend == "s3":
            return FileStorage.s3(
                bucket        = os.getenv("S3_BUCKET", "docmind"),
                region        = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1"),
                access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
                secret_key    = os.getenv("AWS_SECRET_ACCESS_KEY"),
            )

        if backend == "r2":
            return FileStorage.r2(
                account_id = os.getenv("R2_ACCOUNT_ID", ""),
                access_key = os.getenv("R2_ACCESS_KEY", ""),
                secret_key = os.getenv("R2_SECRET_KEY", ""),
                bucket     = os.getenv("R2_BUCKET", "docmind"),
            )

        raise ValueError(f"STORAGE_BACKEND không hợp lệ: '{backend}'. Dùng: local|minio|s3|r2")

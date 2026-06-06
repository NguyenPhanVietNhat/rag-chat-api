"""
auth.py  —  Đăng ký, đăng nhập, JWT middleware cho FastAPI
Lưu trữ: SQLite (thay thế users.json)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
import os
import bcrypt
import time as _time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# ── SQLite setup ──────────────────────────────────────────────────────────────
_DB_PATH = Path(os.getenv("USER_DB_PATH", "./users.db"))
def _get_db_path() -> Path:
    import os
    return Path(os.getenv("USER_DB_PATH", "./users.db"))

@contextmanager
def _get_conn():
    """Context manager trả về connection SQLite, tự đóng sau khi dùng."""
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # truy cập cột bằng tên
    conn.execute("PRAGMA journal_mode=WAL") # WAL mode: đọc/ghi song song an toàn
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Tạo bảng users nếu chưa có. Gọi 1 lần lúc startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL,
                full_name   TEXT NOT NULL DEFAULT '',
                password    TEXT NOT NULL,
                role        TEXT NOT NULL DEFAULT 'user',
                is_approved INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            )
        """)
        # Index tăng tốc tìm kiếm theo username
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """)

# Khởi tạo DB ngay khi import module
init_db()

# ── Helpers tương thích với main.py (dùng dict giống users.json cũ) ───────────
def _row_to_dict(row) -> dict:
    """Chuyển sqlite3.Row → dict giống format users.json cũ."""
    if row is None:
        return {}
    return {
        "user_id":    row["user_id"],
        "username":   row["username"],
        "full_name":  row["full_name"],
        "password":   row["password"],
        "role":       row["role"],
        "is_approved": bool(row["is_approved"]),
        "created_at": row["created_at"],
    }

def _load_users() -> dict:
    """
    Tương thích ngược với code cũ dùng _load_users().
    Trả về dict {username: user_dict} giống users.json.
    """
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
    return {row["username"]: _row_to_dict(row) for row in rows}

def _save_users(users: dict):
    """
    Tương thích ngược — upsert toàn bộ dict vào SQLite.
    Dùng cho các hàm admin cũ vẫn gọi _save_users().
    """
    with _get_conn() as conn:
        for username, u in users.items():
            conn.execute("""
                INSERT INTO users (user_id, username, full_name, password, role, is_approved, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    full_name   = excluded.full_name,
                    password    = excluded.password,
                    role        = excluded.role,
                    is_approved = excluded.is_approved
            """, (
                u["user_id"], u["username"], u.get("full_name", ""),
                u["password"], u.get("role", "user"),
                int(u.get("is_approved", False)),
                u.get("created_at", datetime.now(timezone.utc).isoformat()),
            ))

def _get_user(username: str) -> Optional[dict]:
    """Lấy 1 user theo username, trả về dict hoặc None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return _row_to_dict(row) if row else None

def _count_users() -> int:
    """Đếm tổng số user trong DB."""
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

def _insert_user(u: dict):
    """Thêm user mới vào DB."""
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO users (user_id, username, full_name, password, role, is_approved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            u["user_id"], u["username"], u.get("full_name", ""),
            u["password"], u.get("role", "user"),
            int(u.get("is_approved", False)),
            u.get("created_at", datetime.now(timezone.utc).isoformat()),
        ))

def _update_user_field(username: str, **fields):
    """Cập nhật một hoặc nhiều field của user."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [username]
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE username = ?", values
        )

def _delete_user(username: str):
    """Xoá user khỏi DB."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))


# ── Password hashing (bcrypt ) ───────────────────────────────────
def _hash_password(password: str, salt: str = "") -> str:
    """Hash password bằng bcrypt. Tham số salt giữ để tương thích API cũ nhưng không dùng."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify_password(plain: str, stored: str) -> bool:
    try:
        # Tương thích ngược: nếu password cũ dùng SHA256 (có dấu ":") thì vẫn verify được
        if ":" in stored and not stored.startswith("$2"):
            salt, hashed = stored.split(":", 1)
            return hmac.compare_digest(
                hashlib.sha256(f"{salt}{plain}".encode()).hexdigest(),
                hashed,
            )
        # Password mới dùng bcrypt
        return bcrypt.checkpw(plain.encode(), stored.encode())
    except Exception:
        return False


# ── JWT ───────────────────────────────────────────────────────────────────────
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))

def create_access_token(user_id: str, username: str, role: str = "user") -> str:
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    exp     = int(_time.time()) + config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload = _b64url(json.dumps({"sub": user_id, "username": username, "role": role, "exp": exp}).encode())
    sig_input = f"{header}.{payload}".encode()
    sig = _b64url(
        hmac.new(config.SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{sig}"

def verify_token(token: str) -> dict:
    try:
        header, payload, sig = token.split(".")
        sig_input = f"{header}.{payload}".encode()
        expected = _b64url(
            hmac.new(config.SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        data = json.loads(_b64url_decode(payload))
        if data.get("exp", 0) < _time.time():
            raise ValueError("token expired")
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token không hợp lệ: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── OAuth2 / Dependencies ─────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return verify_token(token)

def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chỉ admin mới có quyền thực hiện thao tác này",
        )
    return current_user


# ── Pydantic models ───────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username:  str
    password:  str
    full_name: str = ""

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str  = "bearer"
    user_id:      str
    username:     str
    role:         str  = "user"
    is_approved:  bool = False


# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse, summary="Đăng ký tài khoản mới")
def register(req: RegisterRequest):
    if _get_user(req.username):
        raise HTTPException(status_code=400, detail="Username đã tồn tại")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Mật khẩu phải ít nhất 6 ký tự")

    is_first  = _count_users() == 0
    role      = "admin" if is_first else "user"
    approved  = is_first

    user = {
        "user_id":    str(uuid.uuid4())[:8],
        "username":   req.username,
        "full_name":  req.full_name,
        "password":   _hash_password(req.password),
        "role":       role,
        "is_approved": approved,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _insert_user(user)

    token = create_access_token(user["user_id"], req.username, role)
    return TokenResponse(
        access_token=token,
        user_id=user["user_id"],
        username=req.username,
        role=role,
        is_approved=approved,
    )

@router.post("/login", response_model=TokenResponse, summary="Đăng nhập")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = _get_user(form.username)
    if not user or not _verify_password(form.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tài khoản hoặc mật khẩu",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.get("is_approved", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tài khoản chưa được admin phê duyệt",
        )
    token = create_access_token(user["user_id"], user["username"], user["role"])
    return TokenResponse(
        access_token=token,
        user_id=user["user_id"],
        username=user["username"],
        role=user["role"],
        is_approved=user["is_approved"],
    )

@router.get("/me", summary="Thông tin tài khoản hiện tại")
def me(current_user: dict = Depends(get_current_user)):
    user = _get_user(current_user.get("username", "")) or {}
    return {
        "user_id":    current_user["sub"],
        "username":   current_user.get("username", ""),
        "full_name":  user.get("full_name", ""),
        "role":       user.get("role", "user"),
        "is_approved": user.get("is_approved", True),
        "created_at": user.get("created_at", ""),
    }

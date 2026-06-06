"""
session_store.py — Lưu trữ lịch sử chat vào SQLite
Thay thế dict RAM trong RAGAgent._sessions
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


_DB_PATH = Path("./users.db")   # dùng chung file DB với auth.py


@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_chat_tables():
    """Tạo bảng chat nếu chưa có. Gọi lúc startup."""
    with _get_conn() as conn:
        # Bảng sessions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id  TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT 'Cuộc hội thoại mới',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON chat_sessions(user_id)
        """)

        # Bảng messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                ts          TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON chat_messages(session_id)
        """)


# Khởi tạo ngay khi import
init_chat_tables()


# ── Session CRUD ──────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str, user_id: str) -> dict:
    """Lấy session từ DB, tạo mới nếu chưa có. Trả về dict session."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if not row:
            conn.execute("""
                INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, "Cuộc hội thoại mới", now, now))
            return {
                "session_id": session_id,
                "user_id":    user_id,
                "title":      "Cuộc hội thoại mới",
                "created_at": now,
                "updated_at": now,
            }

        return dict(row)


def update_session_title(session_id: str, title: str):
    """Cập nhật tiêu đề session (lấy từ tin nhắn đầu tiên)."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute("""
            UPDATE chat_sessions SET title = ?, updated_at = ?
            WHERE session_id = ?
        """, (title[:60], now, session_id))


def touch_session(session_id: str):
    """Cập nhật updated_at của session."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )


def list_sessions(user_id: str) -> list[dict]:
    """Danh sách sessions của 1 user, sắp xếp mới nhất trước."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT s.session_id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON s.session_id = m.session_id
            WHERE s.user_id = ?
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        """, (user_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    """Xoá session và toàn bộ messages (CASCADE)."""
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
        )


# ── Message CRUD ──────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str) -> dict:
    """Thêm 1 tin nhắn vào session. Trả về dict message."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO chat_messages (session_id, role, content, ts)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, now))
    touch_session(session_id)
    return {"session_id": session_id, "role": role, "content": content, "ts": now}


def get_messages(session_id: str, limit: int = 40) -> list[dict]:
    """
    Lấy N tin nhắn gần nhất của session (để đưa vào context OpenAI).
    limit=40 tương đương MAX_TURNS*2 mặc định.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT role, content, ts FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()
    # Đảo ngược để thứ tự cũ → mới
    return [dict(r) for r in reversed(rows)]


def get_all_messages(session_id: str) -> list[dict]:
    """Lấy toàn bộ tin nhắn (dùng cho frontend hiển thị lịch sử)."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT role, content, ts FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,)).fetchall()
    return [dict(r) for r in rows]


def messages_for_openai(session_id: str, max_turns: int = 20) -> list[dict]:
    """
    Trả về list messages theo format OpenAI API:
    [{"role": "user"|"assistant", "content": "..."}]
    Chỉ lấy max_turns*2 messages gần nhất để không vượt context window.
    """
    msgs = get_messages(session_id, limit=max_turns * 2)
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

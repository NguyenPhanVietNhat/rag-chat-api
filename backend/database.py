"""
database.py
───────────
PostgreSQL + SQLAlchemy 2.0 (async) cho hệ thống RAG.
Lưu: users, documents metadata, chat sessions, chat messages.

Dùng SQLite khi dev local (không cần cài PostgreSQL):
    DATABASE_URL=sqlite+aiosqlite:///./docmind.db

Dùng PostgreSQL khi production:
    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/docmind

Chạy migration lần đầu:
    python database.py          ← tự tạo tất cả bảng
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
    func, select, update, delete,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ── URL: ưu tiên .env, fallback SQLite để dev local ──────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./docmind.db",   # dev local
    # "postgresql+asyncpg://postgres:postgres@localhost:5432/docmind",  # production
)

# ── Engine + Session factory ──────────────────────────────────────────────────
_engine = create_async_engine(
    DATABASE_URL,
    echo=False,           # True để log SQL khi debug
    pool_pre_ping=True,   # kiểm tra kết nối trước khi dùng
    # pool_size=10,       # PostgreSQL: số kết nối tối đa
    # max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Base ──────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class User(Base):
    """Người dùng hệ thống."""
    __tablename__ = "users"

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:    Mapped[str]  = mapped_column(String(16), unique=True, nullable=False, index=True)
    username:   Mapped[str]  = mapped_column(String(64), unique=True, nullable=False)
    full_name:  Mapped[str]  = mapped_column(String(128), default="")
    password:   Mapped[str]  = mapped_column(String(128), nullable=False)  # hashed
    is_active:  Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions:  Mapped[list["ChatSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class Document(Base):
    """Metadata tài liệu đã upload và index."""
    __tablename__ = "documents"

    id:          Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id:      Mapped[str]  = mapped_column(String(32), unique=True, nullable=False, index=True)
    user_id:     Mapped[str]  = mapped_column(String(16), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    filename:    Mapped[str]  = mapped_column(String(256), nullable=False)
    doc_type:    Mapped[str]  = mapped_column(String(16), default="")    # pdf, docx, xlsx, txt
    file_size:   Mapped[int]  = mapped_column(Integer, default=0)        # bytes
    chunk_count: Mapped[int]  = mapped_column(Integer, default=0)
    token_count: Mapped[int]  = mapped_column(Integer, default=0)
    page_count:  Mapped[int]  = mapped_column(Integer, default=0)
    status:      Mapped[str]  = mapped_column(String(16), default="indexing")  # indexing | ready | error
    error_msg:   Mapped[str]  = mapped_column(Text, default="")
    query_count: Mapped[int]  = mapped_column(Integer, default=0)        # số lượt được truy vấn
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document {self.filename} [{self.status}]>"


class ChatSession(Base):
    """Một phiên hội thoại (nhiều lượt Q&A)."""
    __tablename__ = "chat_sessions"

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str]  = mapped_column(String(64), nullable=False, index=True)
    user_id:    Mapped[str]  = mapped_column(String(16), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    title:      Mapped[str]  = mapped_column(String(256), default="Cuộc trò chuyện mới")
    turn_count: Mapped[int]  = mapped_column(Integer, default=0)
    is_active:  Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("session_id", "user_id", name="uq_session_user"),
    )

    # Relationships
    user:     Mapped["User"]          = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.id")

    def __repr__(self) -> str:
        return f"<ChatSession {self.session_id} turns={self.turn_count}>"


class ChatMessage(Base):
    """Một lượt Q&A trong session."""
    __tablename__ = "chat_messages"

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str]  = mapped_column(String(64), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id:    Mapped[str]  = mapped_column(String(16), nullable=False)
    role:       Mapped[str]  = mapped_column(String(16), nullable=False)   # user | assistant
    content:    Mapped[str]  = mapped_column(Text, nullable=False)
    # Metadata câu trả lời của AI
    sources:    Mapped[str]  = mapped_column(Text, default="[]")           # JSON list[{file, page, score}]
    tool_calls: Mapped[str]  = mapped_column(String(256), default="[]")    # JSON list[str]
    retrieved:  Mapped[int]  = mapped_column(Integer, default=0)
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    is_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session: Mapped["ChatSession"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage {self.role} len={len(self.content)}>"


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION DEPENDENCY  (dùng trong FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: inject DB session vào endpoint.

    Dùng:
        @app.post("/chat")
        async def chat(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Users ─────────────────────────────────────────────────────────────────────

async def create_user(
    db:        AsyncSession,
    user_id:   str,
    username:  str,
    password:  str,
    full_name: str = "",
) -> User:
    user = User(
        user_id   = user_id,
        username  = username,
        password  = password,
        full_name = full_name,
    )
    db.add(user)
    await db.flush()
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.user_id == user_id))
    return result.scalar_one_or_none()


# ── Documents ─────────────────────────────────────────────────────────────────

async def create_document(
    db:          AsyncSession,
    doc_id:      str,
    user_id:     str,
    filename:    str,
    doc_type:    str = "",
    file_size:   int = 0,
    chunk_count: int = 0,
    token_count: int = 0,
    page_count:  int = 0,
) -> Document:
    doc = Document(
        doc_id      = doc_id,
        user_id     = user_id,
        filename    = filename,
        doc_type    = doc_type,
        file_size   = file_size,
        chunk_count = chunk_count,
        token_count = token_count,
        page_count  = page_count,
        status      = "ready",
    )
    db.add(doc)
    await db.flush()
    return doc


async def get_documents(db: AsyncSession, user_id: str) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document(db: AsyncSession, doc_id: str) -> Optional[Document]:
    result = await db.execute(select(Document).where(Document.doc_id == doc_id))
    return result.scalar_one_or_none()


async def update_document_status(
    db:     AsyncSession,
    doc_id: str,
    status: str,
    error:  str = "",
) -> None:
    await db.execute(
        update(Document)
        .where(Document.doc_id == doc_id)
        .values(status=status, error_msg=error)
    )


async def increment_doc_query_count(db: AsyncSession, doc_id: str) -> None:
    """Tăng counter mỗi khi tài liệu này được truy vấn."""
    await db.execute(
        update(Document)
        .where(Document.doc_id == doc_id)
        .values(query_count=Document.query_count + 1)
    )


async def delete_document(db: AsyncSession, doc_id: str) -> bool:
    result = await db.execute(
        delete(Document).where(Document.doc_id == doc_id)
    )
    return result.rowcount > 0


# ── Chat Sessions ─────────────────────────────────────────────────────────────

async def get_or_create_session(
    db:         AsyncSession,
    session_id: str,
    user_id:    str,
    title:      str = "Cuộc trò chuyện mới",
) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id    == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        session = ChatSession(
            session_id = session_id,
            user_id    = user_id,
            title      = title,
        )
        db.add(session)
        await db.flush()
    return session


async def get_sessions(db: AsyncSession, user_id: str) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id, ChatSession.is_active == True)
        .order_by(ChatSession.updated_at.desc())
    )
    return list(result.scalars().all())


async def delete_session(db: AsyncSession, session_id: str, user_id: str) -> bool:
    result = await db.execute(
        delete(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id    == user_id,
        )
    )
    return result.rowcount > 0


async def auto_title_session(db: AsyncSession, session_id: str, first_question: str) -> None:
    """Tự đặt title = 50 ký tự đầu của câu hỏi đầu tiên."""
    title = first_question[:50] + ("…" if len(first_question) > 50 else "")
    await db.execute(
        update(ChatSession)
        .where(ChatSession.session_id == session_id)
        .values(title=title)
    )


# ── Chat Messages ─────────────────────────────────────────────────────────────

import json as _json

async def save_message(
    db:          AsyncSession,
    session_id:  str,
    user_id:     str,
    role:        str,
    content:     str,
    sources:     list[dict] | None = None,
    tool_calls:  list[str]  | None = None,
    retrieved:   int               = 0,
    elapsed_ms:  float             = 0.0,
    is_fallback: bool              = False,
) -> ChatMessage:
    msg = ChatMessage(
        session_id  = session_id,
        user_id     = user_id,
        role        = role,
        content     = content,
        sources     = _json.dumps(sources or [],   ensure_ascii=False),
        tool_calls  = _json.dumps(tool_calls or [], ensure_ascii=False),
        retrieved   = retrieved,
        elapsed_ms  = elapsed_ms,
        is_fallback = is_fallback,
    )
    db.add(msg)

    # Tăng turn count
    await db.execute(
        update(ChatSession)
        .where(ChatSession.session_id == session_id)
        .values(turn_count=ChatSession.turn_count + 1)
    )
    await db.flush()
    return msg


async def get_messages(
    db:         AsyncSession,
    session_id: str,
    limit:      int = 100,
) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_recent_messages(
    db:         AsyncSession,
    session_id: str,
    n_turns:    int = 10,
) -> list[dict]:
    """
    Lấy N lượt gần nhất dạng list[{role, content}]
    → đưa thẳng vào Claude messages history.
    """
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(n_turns * 2)   # mỗi turn có user + assistant
    )
    msgs = list(reversed(result.scalars().all()))
    return [{"role": m.role, "content": m.content} for m in msgs]


# ── Stats ─────────────────────────────────────────────────────────────────────

async def get_user_stats(db: AsyncSession, user_id: str) -> dict:
    """Dashboard stats cho một user."""
    from sqlalchemy import func as f

    # Tổng tài liệu
    doc_count = (await db.execute(
        select(f.count()).where(Document.user_id == user_id)
        .select_from(Document)
    )).scalar_one()

    # Tổng chunks
    chunk_sum = (await db.execute(
        select(f.sum(Document.chunk_count)).where(Document.user_id == user_id)
    )).scalar_one() or 0

    # Tổng session
    session_count = (await db.execute(
        select(f.count()).where(ChatSession.user_id == user_id)
        .select_from(ChatSession)
    )).scalar_one()

    # Tổng lượt hỏi (chỉ đếm role=user)
    msg_count = (await db.execute(
        select(f.count()).where(
            ChatMessage.user_id == user_id,
            ChatMessage.role    == "user",
        ).select_from(ChatMessage)
    )).scalar_one()

    # Tài liệu được query nhiều nhất
    top_docs_result = await db.execute(
        select(Document.filename, Document.query_count)
        .where(Document.user_id == user_id)
        .order_by(Document.query_count.desc())
        .limit(5)
    )
    top_docs = [{"filename": r[0], "query_count": r[1]} for r in top_docs_result]

    return {
        "doc_count":     doc_count,
        "chunk_count":   chunk_sum,
        "session_count": session_count,
        "message_count": msg_count,
        "top_documents": top_docs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DB INIT  (tạo bảng nếu chưa có)
# ═══════════════════════════════════════════════════════════════════════════════

async def init_db() -> None:
    """Tạo tất cả bảng. Gọi khi khởi động ứng dụng."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"✅ Database khởi tạo xong: {DATABASE_URL.split('://')[0]}")


async def drop_all() -> None:
    """Xoá tất cả bảng (chỉ dùng khi test)."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ── Chạy trực tiếp để khởi tạo DB ────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())

"""
main.py  —  FastAPI application hoàn chỉnh
═══════════════════════════════════════════
Tích hợp toàn bộ: Auth + Document Processor + Embedding + RAG Agent

Chạy:
    uvicorn main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from auth import get_current_user, get_current_admin, router as auth_router
import session_store

# ── Lazy imports (tránh crash khi chưa cài thư viện AI) ──────────────────────
try:
    from document_processor import DocumentProcessor
    from embedding_engine   import EmbeddingEngine, VectorStore
    from rag_agent          import RAGAgent
    _AI_READY = True
except ImportError as _e:
    _AI_READY = False
    print(f"⚠️  AI modules chưa sẵn sàng: {_e}")

# ── Singleton instances ───────────────────────────────────────────────────────
_store:     Optional[object] = None
_agent:     Optional[object] = None
_processor: Optional[object] = None

def get_store():
    global _store
    if _store is None and _AI_READY:
        engine  = EmbeddingEngine(config.EMBED_MODEL)
        _store  = VectorStore(engine=engine, persist_dir=config.CHROMA_DIR)
    return _store

def get_agent():
    global _agent
    if _agent is None:
        _agent = RAGAgent(
            vector_store = get_store(),
            api_key      = config.OPENAI_API_KEY,
            top_k        = config.TOP_K,
            min_score    = config.MIN_SCORE,
        )
    return _agent

def get_processor():
    global _processor
    if _processor is None and _AI_READY:
        _processor = DocumentProcessor(
            chunk_size    = config.CHUNK_SIZE,
            chunk_overlap = config.CHUNK_OVERLAP,
        )
    return _processor

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "DocMind RAG API",
    description = "Hỏi đáp tài liệu nội bộ — FastAPI + Claude AI + sentence-transformers",
    version     = "1.0.0",
    docs_url    = "/api/docs",   # đổi từ /docs → /api/docs để tránh xung đột với endpoint tài liệu
    redoc_url   = "/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = config.ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve frontend tĩnh (tùy chọn)
_frontend = Path("./frontend")
if _frontend.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend), html=True), name="frontend")

# Gắn auth router
app.include_router(auth_router)

# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer:      str
    sources:     list[dict]
    retrieved:   int
    elapsed_ms:  float
    tool_calls:  list[str]
    is_fallback: bool
    session_id:  str

class UploadResponse(BaseModel):
    doc_id:       str
    source_file:  str
    chunks_added: int
    token_count:  int
    message:      str

# ── CHAT ─────────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model = ChatResponse,
    summary        = "Hỏi đáp tài liệu (blocking)",
    tags           = ["Chat"],
)
async def chat(
    req:          ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Gửi câu hỏi → nhận câu trả lời đầy đủ kèm nguồn trích dẫn.
    Yêu cầu: `Authorization: Bearer <token>`
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    # Kiểm tra tài khoản đã được duyệt chưa
    if not current_user.get("is_approved", True) and current_user.get("role") != "admin":
        from auth import _load_users
        _u = _load_users().get(current_user.get("username",""), {})
        if not _u.get("is_approved", False):
            raise HTTPException(status_code=403, detail="Tài khoản chưa được admin phê duyệt")
    user_id = current_user["sub"]
    agent   = get_agent()

    try:
        reply = await asyncio.to_thread(
            agent.chat,
            question   = req.question,
            user_id    = user_id,
            session_id = f"{user_id}:{req.session_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Agent: {e}")

    return ChatResponse(
        answer      = reply.answer,
        sources     = [
            {
                "file":    s.file,
                "page":    s.page,
                "section": s.section,
                "score":   s.score,
                "excerpt": s.excerpt,
            }
            for s in reply.sources
        ],
        retrieved   = reply.retrieved,
        elapsed_ms  = round(reply.elapsed_ms, 1),
        tool_calls  = reply.tool_calls,
        is_fallback = reply.is_fallback,
        session_id  = req.session_id,
    )


@app.post(
    "/chat/stream",
    summary = "Hỏi đáp streaming (Server-Sent Events)",
    tags    = ["Chat"],
)
async def chat_stream(
    req:          ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Streaming version — client nhận từng token ngay khi Claude sinh ra.

    SSE format:
        data: {"type": "token", "text": "Theo "}
        data: {"type": "done"}
        data: {"type": "error", "message": "..."}
    """
    user_id = current_user["sub"]
    agent   = get_agent()

    async def event_gen():
        try:
            loop = asyncio.get_event_loop()
            q    = asyncio.Queue()

            def run_stream():
                try:
                    for token in agent.stream_chat(
                        question   = req.question,
                        user_id    = user_id,
                        session_id = f"{user_id}:{req.session_id}",
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, ("token", token))
                    loop.call_soon_threadsafe(q.put_nowait, ("done", None))
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

            threading.Thread(target=run_stream, daemon=True).start()

            while True:
                kind, payload = await q.get()
                if kind == "token":
                    yield f"data: {json.dumps({'type':'token','text':payload}, ensure_ascii=False)}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'type':'done'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type':'error','message':payload}, ensure_ascii=False)}\n\n"
                    break
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── SESSIONS ──────────────────────────────────────────────────────────────────

@app.get("/sessions", summary="Danh sách sessions", tags=["Chat"])
def list_sessions(current_user: dict = Depends(get_current_user)):
    user_id  = current_user["sub"]
    sessions = session_store.list_sessions(user_id)
    # Cắt bỏ prefix "userId:" trả về short ID cho frontend
    result = []
    for s in sessions:
        short_id = s["session_id"].split(":", 1)[1] if ":" in s["session_id"] else s["session_id"]
        result.append({**s, "session_id": short_id})
    return {"user_id": user_id, "sessions": result}

@app.delete("/sessions/{session_id}", summary="Xoá session", tags=["Chat"])
def delete_session(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    agent    = get_agent()
    full_sid = f"{current_user['sub']}:{session_id}"
    agent.clear_session(full_sid)
    return {"message": f"Đã xoá session '{session_id}'"}

@app.get("/sessions/{session_id}/messages", summary="Lịch sử chat của session", tags=["Chat"])
def get_session_messages(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    """Trả về toàn bộ tin nhắn của 1 session. Frontend dùng khi user mở lại lịch sử."""
    user_id  = current_user["sub"]
    full_sid = f"{user_id}:{session_id}"
 
    sessions = session_store.list_sessions(user_id)
    owned    = any(s["session_id"] == full_sid for s in sessions)
    if not owned:
        raise HTTPException(status_code=404, detail="Session không tồn tại")
 
    messages = session_store.get_all_messages(full_sid)
    return {
        "session_id": session_id,
        "messages":   messages,
    }
 
# ── DOCUMENTS ─────────────────────────────────────────────────────────────────

ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt", ".md"}

@app.post(
    "/docs/upload",
    response_model = UploadResponse,
    summary        = "Upload và index tài liệu",
    tags           = ["Documents"],
)
async def upload_document(
    file:         UploadFile = File(...),
    current_user: dict       = Depends(get_current_user),
):
    """
    Upload file → DocumentProcessor (chunk) → EmbeddingEngine → VectorStore.
    Hỗ trợ: PDF, DOCX, XLSX, TXT (tối đa 50 MB).
    """
    if not _AI_READY:
        raise HTTPException(status_code=501, detail="AI modules chưa được cài đặt")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Định dạng '{suffix}' không hỗ trợ")

    contents = await file.read()
    if len(contents) > config.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File quá lớn (tối đa {config.MAX_FILE_MB} MB)")

    user_id  = current_user["sub"]
    tmp_path = config.UPLOAD_DIR / f"{user_id}_{file.filename}"
    tmp_path.write_bytes(contents)

    try:
        proc   = get_processor()
        result = proc.process(tmp_path, user_id=user_id)

        if not result.ok:
            raise HTTPException(status_code=422, detail=f"Không đọc được nội dung: {result.errors}")

        store = get_store()
        added = store.add_chunks(result.chunks)

        return UploadResponse(
            doc_id       = result.doc_id,
            source_file  = result.source_file,
            chunks_added = added,
            token_count  = result.token_count,
            message      = f"✅ Đã index {added} chunks từ '{file.filename}'",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs", summary="Danh sách tài liệu", tags=["Documents"])
def list_documents(current_user: dict = Depends(get_current_user)):
    """Trả về tất cả tài liệu đã index của user hiện tại."""
    store   = get_store()
    user_id = current_user["sub"]
    if store is None:
        return {"documents": []}

    try:
        data  = store._col.get(
            where   = {"user_id": {"$eq": user_id}},
            include = ["metadatas"],
        )
        files: dict[str, dict] = {}
        for meta in data.get("metadatas", []):
            fname = meta.get("source_file", "")
            if fname and fname not in files:
                files[fname] = {
                    "source_file": fname,
                    "doc_id":      meta.get("doc_id", ""),
                    "doc_type":    meta.get("doc_type", ""),
                    "created_at":  meta.get("created_at", ""),
                }
        return {"user_id": user_id, "count": len(files), "documents": list(files.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/docs/{doc_id}", summary="Xoá tài liệu", tags=["Documents"])
def delete_document(
    doc_id:       str,
    current_user: dict = Depends(get_current_user),
):
    store   = get_store()
    if store is None:
        raise HTTPException(status_code=501, detail="VectorStore chưa khởi tạo")
    deleted = store.delete_doc(doc_id)
    return {"message": f"Đã xoá {deleted} chunks của doc_id={doc_id}"}


# ── ADMIN ENDPOINTS ───────────────────────────────────────────────────────────

@app.get("/admin/users", summary="[Admin] Danh sách tất cả user", tags=["Admin"])
def admin_list_users(current_admin: dict = Depends(get_current_admin)):
    """Admin xem toàn bộ danh sách user và trạng thái."""
    from auth import _load_users
    users = _load_users()
    result = []
    for uname, u in users.items():
        result.append({
            "username":    uname,
            "user_id":     u.get("user_id", ""),
            "full_name":   u.get("full_name", ""),
            "role":        u.get("role", "user"),
            "is_approved": u.get("is_approved", False),
            "created_at":  u.get("created_at", ""),
        })
    return {"total": len(result), "users": result}


@app.post("/admin/users/{username}/approve", summary="[Admin] Phê duyệt user", tags=["Admin"])
def admin_approve_user(
    username:      str,
    current_admin: dict = Depends(get_current_admin),
):
    """Admin phê duyệt tài khoản user để cho phép đăng nhập."""
    from auth import _load_users, _save_users
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User không tồn tại")
    users[username]["is_approved"] = True
    _save_users(users)
    return {"message": f"Đã phê duyệt user: {username}"}


@app.post("/admin/users/{username}/revoke", summary="[Admin] Thu hồi quyền user", tags=["Admin"])
def admin_revoke_user(
    username:      str,
    current_admin: dict = Depends(get_current_admin),
):
    """Admin thu hồi quyền truy cập của user."""
    from auth import _load_users, _save_users
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User không tồn tại")
    if users[username].get("role") == "admin":
        raise HTTPException(status_code=400, detail="Không thể thu hồi quyền admin")
    users[username]["is_approved"] = False
    _save_users(users)
    return {"message": f"Đã thu hồi quyền: {username}"}


@app.delete("/admin/users/{username}", summary="[Admin] Xoá user", tags=["Admin"])
def admin_delete_user(
    username:      str,
    current_admin: dict = Depends(get_current_admin),
):
    """Admin xoá tài khoản user."""
    from auth import _load_users,  _delete_user
    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User không tồn tại")
    if users[username].get("role") == "admin":
        raise HTTPException(status_code=400, detail="Không thể xoá tài khoản admin")
    _delete_user(username)
    return {"message": f"Đã xoá user: {username}"}


@app.get("/admin/docs", summary="[Admin] Tất cả tài liệu hệ thống", tags=["Admin"])
def admin_list_all_docs(current_admin: dict = Depends(get_current_admin)):
    """Admin xem toàn bộ tài liệu của tất cả user."""
    store = get_store()
    if store is None:
        return {"documents": []}
    try:
        data  = store._col.get(include=["metadatas"])
        files: dict[str, dict] = {}
        for meta in data.get("metadatas", []):
            fname = meta.get("source_file", "")
            key   = f"{meta.get('user_id','')}::{fname}"
            if fname and key not in files:
                files[key] = {
                    "source_file": fname,
                    "doc_id":      meta.get("doc_id", ""),
                    "doc_type":    meta.get("doc_type", ""),
                    "user_id":     meta.get("user_id", ""),
                    "created_at":  meta.get("created_at", ""),
                }
        return {"count": len(files), "documents": list(files.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/docs/{doc_id}", summary="[Admin] Xoá tài liệu bất kỳ", tags=["Admin"])
def admin_delete_doc(
    doc_id:        str,
    current_admin: dict = Depends(get_current_admin),
):
    """Admin xoá tài liệu của bất kỳ user nào."""
    store = get_store()
    if store is None:
        raise HTTPException(status_code=501, detail="VectorStore chưa khởi tạo")
    deleted = store.delete_doc(doc_id)
    return {"message": f"Admin đã xoá {deleted} chunks của doc_id={doc_id}"}


@app.post("/admin/users/{username}/role", summary="[Admin] Đổi role user", tags=["Admin"])
def admin_change_role(
    username:      str,
    current_admin: dict = Depends(get_current_admin),
):
    """
    Admin đổi role của user: user ↔ admin.
    Không thể tự đổi role của chính mình.
    """
    from auth import _load_users, _save_users

    if username == current_admin.get("username"):
        raise HTTPException(status_code=400, detail="Không thể tự đổi role của chính mình")

    users = _load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User không tồn tại")

    current_role = users[username].get("role", "user")
    new_role     = "admin" if current_role == "user" else "user"

    users[username]["role"]        = new_role
    users[username]["is_approved"] = True   # admin luôn được approved
    _save_users(users)

    return {
        "message":  f"Đã đổi role @{username}: {current_role} → {new_role}",
        "username": username,
        "new_role": new_role,
    }


# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Trạng thái hệ thống")
def health():
    store = get_store()
    return {
        "status":      "ok",
        "ai_ready":    _AI_READY,
        "api_key_set": bool(config.OPENAI_API_KEY),
        "embed_model": config.EMBED_MODEL,
        "vector_store": store.stats() if store else None,
        "chunk_config": {
            "size":     config.CHUNK_SIZE,
            "overlap":  config.CHUNK_OVERLAP,
            "top_k":    config.TOP_K,
            "min_score": config.MIN_SCORE,
        },
    }

@app.get("/", include_in_schema=False)
def root():
    return {"message": "DocMind RAG API — truy cập /docs để xem Swagger UI"}

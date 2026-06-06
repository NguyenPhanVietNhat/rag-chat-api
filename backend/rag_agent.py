"""
rag_agent.py
────────────
AI Agent trả lời câu hỏi dựa trên ngữ cảnh RAG với OpenAI API.

Tính năng:
  - Tích hợp VectorStore.search() → build_context() → GPT-4o
  - Conversation memory (multi-turn)
  - Streaming response (SSE)
  - Tool use: search_documents, get_document_list, clarify_question
  - Guard rails: từ chối câu hỏi ngoài phạm vi tài liệu
  - Citation: trích dẫn nguồn kèm câu trả lời
  - Fallback graceful khi không tìm thấy thông tin

Cách dùng:
    agent = RAGAgent(vector_store=store)
    reply = agent.chat("Điều khoản thanh toán quy định thế nào?", user_id="u1")
    print(reply.answer)
    print(reply.sources)
"""

from __future__ import annotations

import json
import logging
import time
import session_store
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── OpenAI model ──────────────────────────────────────────────────────────────
OPENAI_MODEL   = "gpt-4o"          # hoặc "gpt-4o-mini" để tiết kiệm chi phí
MAX_TOKENS     = 1024
MAX_TURNS      = 20                # Tối đa số lượt trong một session
TOP_K_CHUNKS   = 5                 # Số chunks retrieve mỗi lần search
MIN_SCORE      = 0.20              # Ngưỡng similarity tối thiểu


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Source:
    """Một nguồn tài liệu được trích dẫn trong câu trả lời."""
    file:    str
    page:    Optional[int]
    section: Optional[str]
    score:   float
    excerpt: str            # đoạn văn ngắn làm bằng chứng

    def label(self) -> str:
        loc = f", trang {self.page}" if self.page else (f", {self.section}" if self.section else "")
        return f"{self.file}{loc}"


@dataclass
class AgentReply:
    """Kết quả trả về sau mỗi lượt chat."""
    answer:      str
    sources:     list[Source]
    retrieved:   int           # số chunks đã retrieve
    elapsed_ms:  float
    tool_calls:  list[str]     # tên các tool đã gọi
    is_fallback: bool = False  # True nếu không tìm thấy thông tin


@dataclass
class Turn:
    """Một lượt hội thoại."""
    role:    str     # "user" | "assistant" | "tool"
    content: str
    ts:      str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Session:
    """Session hội thoại của một user."""
    session_id: str
    user_id:    str
    turns:      list[Turn] = field(default_factory=list)
    created_at: str        = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add(self, role: str, content: str):
        self.turns.append(Turn(role=role, content=content))

    def messages_for_api(self) -> list[dict]:
        """Chuyển turns → format messages cho OpenAI API."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def trim(self, max_turns: int = MAX_TURNS):
        """Giữ N turns gần nhất để tránh vượt context window."""
        if len(self.turns) > max_turns * 2:
            self.turns = self.turns[-(max_turns * 2):]


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên hỏi đáp tài liệu nội bộ của doanh nghiệp.

## Nhiệm vụ
Trả lời câu hỏi của người dùng **chỉ dựa trên nội dung tài liệu được cung cấp** thông qua các tool.
Không được bịa đặt thông tin ngoài tài liệu.

## Quy tắc trả lời
1. **Luôn gọi tool `search_documents` trước** khi trả lời bất kỳ câu hỏi nào về nội dung.
2. Nếu tìm thấy thông tin: trả lời rõ ràng, súc tích, kèm trích dẫn [Nguồn: tên file, trang X].
3. Nếu không tìm thấy: thừa nhận thẳng thắn "Tôi không tìm thấy thông tin này trong tài liệu".
4. Không suy diễn hay phỏng đoán ngoài phạm vi tài liệu.
5. Dùng tiếng Việt, trừ khi người dùng hỏi bằng tiếng Anh.
6. Câu trả lời ngắn gọn (3–5 câu), trừ khi cần liệt kê chi tiết.

## Định dạng trích dẫn
Khi trích dẫn, dùng: **[Nguồn: tên_file.pdf, trang N]** hoặc **[Nguồn: tên_file.docx, mục "Tên mục"]**

## Giới hạn
- Không thảo luận chủ đề ngoài nội dung tài liệu (tin tức, lập trình tổng quát, v.v.)
- Không tiết lộ nội dung tài liệu của user khác
- Nếu câu hỏi mơ hồ, gọi tool `clarify_question` để hỏi lại
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS  (OpenAI function calling format)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name":        "search_documents",
            "description": (
                "Tìm kiếm thông tin liên quan trong kho tài liệu nội bộ. "
                "Gọi tool này TRƯỚC KHI trả lời mọi câu hỏi về nội dung. "
                "Trả về các đoạn văn bản phù hợp nhất kèm nguồn tài liệu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Câu truy vấn tìm kiếm (viết lại rõ ràng, cụ thể hơn câu hỏi gốc nếu cần)",
                    },
                    "top_k": {
                        "type":        "integer",
                        "description": "Số kết quả cần lấy (mặc định 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_document_list",
            "description": "Lấy danh sách tài liệu đã được index trong hệ thống. Dùng khi người dùng hỏi 'có những tài liệu nào' hoặc cần biết phạm vi dữ liệu.",
            "parameters": {
                "type":       "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "clarify_question",
            "description": "Yêu cầu người dùng làm rõ câu hỏi khi quá mơ hồ hoặc có nhiều cách hiểu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clarification": {
                        "type":        "string",
                        "description": "Câu hỏi làm rõ gửi đến người dùng",
                    },
                },
                "required": ["clarification"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# RAG AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class RAGAgent:
    """
    AI Agent hỏi đáp tài liệu dùng OpenAI GPT-4o + Function Calling + RAG.

    Luồng xử lý:
      1. Nhận câu hỏi → thêm vào session history
      2. Gọi OpenAI với tool definitions
      3. GPT-4o quyết định gọi function nào (thường là search_documents)
      4. Agent thực thi tool → trả kết quả cho GPT-4o
      5. GPT-4o tổng hợp → trả lời cuối cùng
      6. Lưu vào session history

    Ví dụ:
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("Điều khoản thanh toán?", user_id="u1", session_id="s1")
        print(reply.answer)
        for src in reply.sources:
            print(src.label(), src.score)
    """

    def __init__(
        self,
        vector_store           = None,
        model:      str        = OPENAI_MODEL,
        top_k:      int        = TOP_K_CHUNKS,
        min_score:  float      = MIN_SCORE,
        api_key:    Optional[str] = None,
    ):
        self.store     = vector_store
        self.model     = model
        self.top_k     = top_k
        self.min_score = min_score
        self.client    = OpenAI(api_key=api_key)  # đọc OPENAI_API_KEY từ env

    # ── Session management ────────────────────────────────────────────────────

    def get_or_create_session(self, session_id: str, user_id: str) -> Session:
        """Tạo session trong DB nếu chưa có, load history vào RAM."""
        session_store.get_or_create_session(session_id, user_id)
        # Load messages từ DB → Session object để dùng trong chat
        session = Session(session_id=session_id, user_id=user_id)
        for m in session_store.get_messages(session_id):
            session.turns.append(Turn(role=m["role"], content=m["content"]))
        return session

    def clear_session(self, session_id: str):
        """Xoá session khỏi DB."""
        session_store.delete_session(session_id)

    def list_sessions(self, user_id: str) -> list[dict]:
        """Danh sách sessions của user từ DB."""
        return session_store.list_sessions(user_id)

    # ── Tool execution ────────────────────────────────────────────────────────

    def _exec_search(self, query: str, user_id: str, top_k: int = None) -> dict:
        if self.store is None:
            return {"error": "VectorStore chưa được khởi tạo"}

        hits = self.store.search(
            query     = query,
            top_k     = top_k or self.top_k,
            user_id   = user_id,
            min_score = self.min_score,
        )

        if not hits:
            return {
                "found":   0,
                "message": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "results": [],
            }

        return {
            "found": len(hits),
            "results": [
                {
                    "rank":    i + 1,
                    "score":   hit.score,
                    "source":  hit.source_file,
                    "page":    hit.page,
                    "section": hit.section,
                    "text":    hit.text,
                }
                for i, hit in enumerate(hits)
            ],
        }

    def _exec_doc_list(self, user_id: str) -> dict:
        if self.store is None:
            return {"error": "VectorStore chưa được khởi tạo"}
        try:
            data  = self.store._col.get(
                where   = {"user_id": {"$eq": user_id}},
                include = ["metadatas"],
            )
            files = {}
            for meta in data.get("metadatas", []):
                fname = meta.get("source_file", "")
                dtype = meta.get("doc_type", "")
                if fname and fname not in files:
                    files[fname] = dtype

            if not files:
                return {"message": "Chưa có tài liệu nào được index.", "documents": []}

            return {
                "count":     len(files),
                "documents": [{"name": f, "type": t} for f, t in files.items()],
            }
        except Exception as e:
            return {"error": str(e)}

    def _dispatch_tool(self, tool_name: str, tool_input: dict, user_id: str) -> str:
        """Thực thi tool → trả về JSON string."""
        if tool_name == "search_documents":
            result = self._exec_search(
                query   = tool_input.get("query", ""),
                user_id = user_id,
                top_k   = tool_input.get("top_k", self.top_k),
            )
        elif tool_name == "get_document_list":
            result = self._exec_doc_list(user_id)
        elif tool_name == "clarify_question":
            result = {"clarification_sent": tool_input.get("clarification", "")}
        else:
            result = {"error": f"Tool '{tool_name}' không tồn tại"}

        return json.dumps(result, ensure_ascii=False)

    # ── Extract sources từ tool results ──────────────────────────────────────

    def _extract_sources(self, tool_results: list[dict]) -> list[Source]:
        sources = []
        seen    = set()
        for tr in tool_results:
            content = tr.get("content", "{}")
            try:
                data = json.loads(content) if isinstance(content, str) else content
            except Exception:
                continue
            for item in data.get("results", []):
                key = (item.get("source", ""), item.get("page"), item.get("section", ""))
                if key in seen:
                    continue
                seen.add(key)
                sources.append(Source(
                    file    = item.get("source", ""),
                    page    = item.get("page"),
                    section = item.get("section"),
                    score   = item.get("score", 0.0),
                    excerpt = item.get("text", "")[:200],
                ))
        return sources

    # ── Main chat ─────────────────────────────────────────────────────────────

    def chat(
        self,
        question:   str,
        user_id:    str = "anonymous",
        session_id: str = "default",
    ) -> AgentReply:
        """
        Gửi câu hỏi → nhận AgentReply.

        Luồng:
          user question
            → GPT-4o (function calling)
            → agent thực thi tool(s)
            → GPT-4o nhận kết quả → sinh câu trả lời
        """
        t0 = time.time()

        session = self.get_or_create_session(session_id, user_id)
        session.add("user", question)
        session_store.add_message(session_id, "user", question)   # <-- THÊM DÒNG NÀY
        # Cập nhật title nếu là tin nhắn đầu tiên
        if len(session.turns) == 1:
            session_store.update_session_title(session_id, question[:50])
        session.trim()

        # OpenAI: system message đứng đầu, rồi mới đến history
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + session.messages_for_api()[:-1]   # history trừ câu hỏi vừa add
            + [{"role": "user", "content": question}]
        )

        tool_calls_made = []
        tool_results    = []
        final_text      = ""
        MAX_LOOPS       = 5

        for _ in range(MAX_LOOPS):
            response = self.client.chat.completions.create(
                model       = self.model,
                max_tokens  = MAX_TOKENS,
                messages    = messages,
                tools       = TOOLS,
                tool_choice = "auto",
            )

            msg        = response.choices[0].message
            finish     = response.choices[0].finish_reason

            # Thêm assistant turn vào messages
            messages.append(msg)

            # Nếu GPT trả lời thẳng (không gọi tool)
            if finish == "stop" or not msg.tool_calls:
                final_text = msg.content or ""
                break

            # Xử lý tool calls
            for tc in msg.tool_calls:
                tool_name  = tc.function.name
                tool_input = json.loads(tc.function.arguments)

                tool_calls_made.append(tool_name)
                result_str = self._dispatch_tool(tool_name, tool_input, user_id)
                tool_results.append({"content": result_str})

                logger.info(f"  Tool: {tool_name}({json.dumps(tool_input, ensure_ascii=False)[:80]})")
                logger.info(f"  Result: {result_str[:120]}…")

                # Trả kết quả tool về cho GPT
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

        # Lưu câu trả lời vào session
        if final_text:
            session.add("assistant", final_text)
            session_store.add_message(session_id, "assistant", final_text)

        # Kiểm tra fallback
        fallback_phrases = ["không tìm thấy", "không có thông tin", "ngoài phạm vi"]
        is_fallback = any(p in final_text.lower() for p in fallback_phrases)

        sources   = self._extract_sources(tool_results)
        elapsed   = (time.time() - t0) * 1000
        retrieved = sum(
            json.loads(tr["content"]).get("found", 0)
            for tr in tool_results
            if tr.get("content")
        )

        logger.info(
            f"RAGAgent: '{question[:50]}' → {len(sources)} sources, "
            f"{len(tool_calls_made)} tools, {elapsed:.0f}ms"
        )

        return AgentReply(
            answer      = final_text or "Tôi không thể tạo câu trả lời lúc này.",
            sources     = sources,
            retrieved   = retrieved,
            elapsed_ms  = elapsed,
            tool_calls  = tool_calls_made,
            is_fallback = is_fallback,
        )

    # ── Streaming chat ────────────────────────────────────────────────────────

    def stream_chat(
        self,
        question:   str,
        user_id:    str = "anonymous",
        session_id: str = "default",
    ) -> Generator[str, None, None]:
        """
        Streaming version: yield từng token text.
        Dùng với FastAPI StreamingResponse / SSE.

        Luồng: tool calls chạy trước (không stream) → stream câu trả lời cuối.
        """
        session = self.get_or_create_session(session_id, user_id)
        session.add("user", question)
        session_store.add_message(session_id, "user", question)
        if len(session.turns) == 1:
            session_store.update_session_title(session_id, question[:50])

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + session.messages_for_api()[:-1]
            + [{"role": "user", "content": question}]
        )

        tool_results = []

        # ── Bước 1: Tool calls (không stream) ─────────────────────────────────
        response = self.client.chat.completions.create(
            model       = self.model,
            max_tokens  = MAX_TOKENS,
            messages    = messages,
            tools       = TOOLS,
            tool_choice = "auto",
        )

        msg    = response.choices[0].message
        finish = response.choices[0].finish_reason
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name  = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                result_str = self._dispatch_tool(tool_name, tool_input, user_id)
                tool_results.append({"content": result_str})
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

            # ── Bước 2: Stream câu trả lời cuối ───────────────────────────────
            full_reply = ""
            stream = self.client.chat.completions.create(
                model      = self.model,
                max_tokens = MAX_TOKENS,
                messages   = messages,
                stream     = True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_reply += delta.content
                    yield delta.content

            session.add("assistant", full_reply)
            session_store.add_message(session_id, "assistant", full_reply)

        else:
            # Không có tool call — yield nội dung trực tiếp
            text = msg.content or ""
            yield text
            session.add("assistant", text)

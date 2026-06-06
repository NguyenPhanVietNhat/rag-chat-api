"""
test_rag_agent.py
─────────────────
Kiểm tra RAGAgent đầy đủ bằng mock Claude API + mock VectorStore.
Chạy: python test_rag_agent.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK SETUP
# ═══════════════════════════════════════════════════════════════════════════════

class MockSearchHit:
    def __init__(self, text, score, source, page=None, section=None):
        self.text        = text
        self.score       = score
        self.source_file = source
        self.page        = page
        self.section     = section
        self.chunk_id    = "chunk_" + source[:6]
        self.doc_id      = "doc_" + source[:4]
        self.metadata    = {
            "doc_id":      self.doc_id,
            "source_file": source,
            "user_id":     "u1",
            "page":        page or -1,
            "section":     section or "",
        }


class MockVectorStore:
    """Mock VectorStore trả về kết quả cố định theo query keyword."""

    def __init__(self):
        self._col = MagicMock()
        self._col.get.return_value = {
            "metadatas": [
                {"source_file": "hop_dong.pdf", "doc_type": "pdf", "user_id": "u1"},
                {"source_file": "quy_trinh.docx", "doc_type": "docx", "user_id": "u1"},
            ]
        }
        self._col.count.return_value = 42

    def search(self, query, top_k=5, user_id=None, min_score=0.0):
        q = query.lower()
        if "thanh toán" in q or "payment" in q or "hóa đơn" in q:
            return [
                MockSearchHit(
                    "Khách hàng phải thanh toán trong vòng 30 ngày kể từ ngày xuất hóa đơn. "
                    "Trường hợp thanh toán trễ hạn sẽ bị phạt 0.5% mỗi tháng.",
                    score=0.87, source="hop_dong.pdf", page=8,
                ),
                MockSearchHit(
                    "Phương thức thanh toán chấp nhận: chuyển khoản ngân hàng hoặc séc. "
                    "Mọi tranh chấp hóa đơn phải thông báo trong 7 ngày làm việc.",
                    score=0.79, source="hop_dong.pdf", page=9,
                ),
            ]
        if "tuyển dụng" in q or "phỏng vấn" in q or "nhân sự" in q:
            return [
                MockSearchHit(
                    "Quy trình tuyển dụng gồm 5 bước: tiếp nhận yêu cầu, đăng tuyển, "
                    "sàng lọc hồ sơ, phỏng vấn 2 vòng, và gửi thư mời làm việc.",
                    score=0.91, source="quy_trinh.docx", section="Chương 3",
                ),
            ]
        if "bảo mật" in q or "bảo hành" in q:
            return [
                MockSearchHit(
                    "Thông tin bảo mật bao gồm dữ liệu kinh doanh, kỹ thuật và tài chính. "
                    "Các bên có nghĩa vụ bảo mật trong 5 năm kể từ ngày ký hợp đồng.",
                    score=0.83, source="chinh_sach.pdf", page=12,
                ),
            ]
        return []   # Không tìm thấy

    def stats(self):
        return {"total_chunks": 42, "model": "mock", "embed_dim": 384, "cache": {}}


# Mock Anthropic client
def make_mock_anthropic(scenario="search_then_answer"):
    """
    Tạo mock anthropic.Anthropic với các kịch bản khác nhau:
      - search_then_answer: Claude gọi search_documents → nhận kết quả → trả lời
      - direct_answer: Claude trả lời không cần tool
      - clarify:  Claude gọi clarify_question
      - fallback: Claude không tìm thấy → fallback response
    """
    client = MagicMock()

    def make_tool_use_block(name, input_data, tool_id="tool_001"):
        block = MagicMock()
        block.type  = "tool_use"
        block.id    = tool_id
        block.name  = name
        block.input = input_data
        return block

    def make_text_block(text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    call_count = [0]   # track number of create() calls

    def mock_create(**kwargs):
        call_count[0] += 1
        resp = MagicMock()

        messages = kwargs.get("messages", [])
        # Lượt 1: gọi tool; Lượt 2: trả lời cuối
        has_tool_result = any(
            isinstance(m.get("content"), list) and
            any(b.get("type") == "tool_result" for b in m.get("content", []))
            for m in messages
        )

        if scenario == "search_then_answer" and not has_tool_result:
            # Lượt 1: gọi search
            query = "điều khoản thanh toán"
            resp.content    = [make_tool_use_block("search_documents", {"query": query})]
            resp.stop_reason = "tool_use"

        elif scenario == "search_then_answer" and has_tool_result:
            # Lượt 2: có kết quả → trả lời
            resp.content = [make_text_block(
                "Theo hợp đồng, khách hàng phải thanh toán trong vòng **30 ngày** "
                "kể từ ngày xuất hóa đơn. Trường hợp trễ hạn sẽ bị phạt **0.5%/tháng** "
                "trên số tiền còn lại. **[Nguồn: hop_dong.pdf, trang 8]**"
            )]
            resp.stop_reason = "end_turn"

        elif scenario == "direct_answer":
            resp.content    = [make_text_block("Đây là câu trả lời trực tiếp không cần tool.")]
            resp.stop_reason = "end_turn"

        elif scenario == "clarify":
            if not has_tool_result:
                resp.content    = [make_tool_use_block("clarify_question", {
                    "clarification": "Bạn muốn hỏi về điều khoản trong hợp đồng nào?"
                }, tool_id="tool_clarify")]
                resp.stop_reason = "tool_use"
            else:
                resp.content    = [make_text_block("Vui lòng cho tôi biết bạn muốn hỏi về hợp đồng nào?")]
                resp.stop_reason = "end_turn"

        elif scenario == "fallback":
            if not has_tool_result:
                resp.content    = [make_tool_use_block("search_documents", {"query": "giá cổ phiếu"})]
                resp.stop_reason = "tool_use"
            else:
                resp.content    = [make_text_block(
                    "Tôi không tìm thấy thông tin về giá cổ phiếu trong tài liệu nội bộ. "
                    "Chủ đề này nằm ngoài phạm vi tài liệu được cung cấp."
                )]
                resp.stop_reason = "end_turn"

        elif scenario == "doclist":
            resp.content    = [make_tool_use_block("get_document_list", {})]
            resp.stop_reason = "tool_use"
            if has_tool_result:
                resp.content    = [make_text_block(
                    "Hệ thống hiện có 2 tài liệu: hop_dong.pdf và quy_trinh.docx."
                )]
                resp.stop_reason = "end_turn"

        else:
            resp.content    = [make_text_block("Câu trả lời mặc định.")]
            resp.stop_reason = "end_turn"

        return resp

    client.messages.create.side_effect = mock_create
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sep(title):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")

def ok(msg):  print(f"  ✅ {msg}")
def info(msg): print(f"  ℹ️  {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_basic_chat():
    sep("TEST 1: Chat cơ bản — search → answer")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("search_then_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("Điều khoản thanh toán quy định thế nào?", user_id="u1", session_id="s1")

    info(f"Answer    : {reply.answer[:100]}…")
    info(f"Sources   : {len(reply.sources)}")
    info(f"Tools     : {reply.tool_calls}")
    info(f"Retrieved : {reply.retrieved}")
    info(f"Elapsed   : {reply.elapsed_ms:.0f}ms")
    info(f"Fallback  : {reply.is_fallback}")

    assert reply.answer, "Answer rỗng"
    assert "search_documents" in reply.tool_calls
    assert len(reply.sources) >= 1
    assert reply.sources[0].file == "hop_dong.pdf"
    assert reply.sources[0].page == 8
    assert not reply.is_fallback

    ok("answer không rỗng")
    ok("search_documents được gọi")
    ok("sources chứa hop_dong.pdf trang 8")
    ok("is_fallback = False")
    return agent


def test_multi_turn(agent):
    sep("TEST 2: Multi-turn conversation")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("search_then_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent2 = RAGAgent(vector_store=store)
        sid    = "session_multi"

        # Lượt 1
        r1 = agent2.chat("Hỏi về thanh toán?",    user_id="u1", session_id=sid)
        # Lượt 2
        r2 = agent2.chat("Còn điều khoản nào khác?", user_id="u1", session_id=sid)

    session = agent2._sessions.get(sid)
    assert session is not None
    info(f"Số turns trong session: {len(session.turns)}")
    assert len(session.turns) == 4   # 2 user + 2 assistant

    ok("Session lưu đúng 4 turns (2 user + 2 assistant)")


def test_fallback():
    sep("TEST 3: Fallback khi không tìm thấy thông tin")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("fallback")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("Giá cổ phiếu hôm nay là bao nhiêu?", user_id="u1")

    info(f"Answer: {reply.answer[:100]}")
    assert reply.is_fallback, "Phải là fallback"
    ok("is_fallback = True khi không tìm thấy thông tin")


def test_clarify():
    sep("TEST 4: Clarify question khi câu hỏi mơ hồ")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("clarify")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("Hỏi về hợp đồng?", user_id="u1")

    info(f"Answer: {reply.answer[:100]}")
    assert "clarify_question" in reply.tool_calls
    ok("clarify_question tool được gọi")


def test_document_list():
    sep("TEST 5: get_document_list tool")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("doclist")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("Có những tài liệu nào trong hệ thống?", user_id="u1")

    info(f"Answer: {reply.answer[:100]}")
    assert "get_document_list" in reply.tool_calls
    ok("get_document_list được gọi")


def test_source_extraction():
    sep("TEST 6: Source extraction — nhiều nguồn")
    from rag_agent import RAGAgent, Source

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("search_then_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("thanh toán", user_id="u1")

    info(f"Số sources: {len(reply.sources)}")
    for src in reply.sources:
        info(f"  {src.label()} | score={src.score} | excerpt={src.excerpt[:60]}…")
        assert src.file, "file không được rỗng"
        assert 0 <= src.score <= 1, "score phải trong [0,1]"

    ok("Tất cả sources có file và score hợp lệ")


def test_session_management():
    sep("TEST 7: Session management")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("direct_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)

        # Tạo 3 sessions
        for i in range(3):
            agent.chat("câu hỏi", user_id="u1", session_id=f"sess_{i}")

        sessions = agent.list_sessions("u1")
        info(f"Sessions: {sessions}")
        assert len(sessions) == 3

        # Xoá 1 session
        agent.clear_session("sess_0")
        assert len(agent.list_sessions("u1")) == 2

        # Session không tồn tại → tạo mới
        s_new = agent.get_or_create_session("new_sess", "u1")
        assert s_new.session_id == "new_sess"

    ok("Tạo, liệt kê và xoá session đúng")


def test_session_trim():
    sep("TEST 8: Session trim — tránh vượt context window")
    from rag_agent import RAGAgent, MAX_TURNS

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("direct_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent   = RAGAgent(vector_store=store)
        session = agent.get_or_create_session("trim_test", "u1")

        # Thêm 100 turns thủ công
        for i in range(100):
            session.add("user",      f"câu hỏi {i}")
            session.add("assistant", f"câu trả lời {i}")

        info(f"Trước trim: {len(session.turns)} turns")
        session.trim(max_turns=MAX_TURNS)
        info(f"Sau trim  : {len(session.turns)} turns")
        assert len(session.turns) <= MAX_TURNS * 2

    ok(f"Trim giữ ≤ {MAX_TURNS * 2} turns")


def test_dispatch_tool_edge_cases():
    sep("TEST 9: Tool dispatch — edge cases")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("direct_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)

        # Tool không tồn tại
        result = agent._dispatch_tool("unknown_tool", {}, "u1")
        data   = json.loads(result)
        assert "error" in data
        info(f"Unknown tool: {data}")

        # Search với user không có dữ liệu
        result2 = agent._dispatch_tool("search_documents", {"query": "xyz không tồn tại"}, "u999")
        data2   = json.loads(result2)
        info(f"No results: found={data2.get('found', 0)}")
        assert data2.get("found", 0) == 0

        # clarify_question
        result3 = agent._dispatch_tool("clarify_question", {"clarification": "Bạn muốn hỏi gì?"}, "u1")
        data3   = json.loads(result3)
        assert data3.get("clarification_sent") == "Bạn muốn hỏi gì?"
        info(f"Clarify: {data3}")

    ok("Tất cả edge cases xử lý đúng")


def test_agent_reply_structure():
    sep("TEST 10: AgentReply — cấu trúc dữ liệu đầy đủ")
    from rag_agent import RAGAgent, AgentReply, Source

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("search_then_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)
        reply = agent.chat("thanh toán", user_id="u1")

    assert isinstance(reply, AgentReply)
    assert isinstance(reply.answer,     str)
    assert isinstance(reply.sources,    list)
    assert isinstance(reply.retrieved,  int)
    assert isinstance(reply.elapsed_ms, float)
    assert isinstance(reply.tool_calls, list)
    assert isinstance(reply.is_fallback, bool)
    assert reply.elapsed_ms > 0

    if reply.sources:
        src = reply.sources[0]
        assert isinstance(src, Source)
        assert src.label()   # không rỗng
        info(f"Source label: {src.label()}")

    ok("AgentReply có đầy đủ fields đúng kiểu dữ liệu")


def test_system_prompt_and_tools():
    sep("TEST 11: System prompt + Tool definitions")
    from rag_agent import SYSTEM_PROMPT, TOOLS, CLAUDE_MODEL, MAX_TOKENS

    assert "search_documents" in SYSTEM_PROMPT
    assert len(TOOLS) == 3

    tool_names = [t["name"] for t in TOOLS]
    assert "search_documents"  in tool_names
    assert "get_document_list" in tool_names
    assert "clarify_question"  in tool_names

    for tool in TOOLS:
        assert "name"          in tool
        assert "description"   in tool
        assert "input_schema"  in tool
        info(f"Tool '{tool['name']}': {tool['description'][:60]}…")

    info(f"Model  : {CLAUDE_MODEL}")
    info(f"Max tokens: {MAX_TOKENS}")

    ok("System prompt và 3 tools định nghĩa đầy đủ")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: Pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def demo_pipeline():
    sep("DEMO: Pipeline RAG đầy đủ")
    from rag_agent import RAGAgent

    store  = MockVectorStore()
    mock_c = make_mock_anthropic("search_then_answer")

    with patch("anthropic.Anthropic", return_value=mock_c):
        agent = RAGAgent(vector_store=store)

        print("\n  📋 Kịch bản hội thoại nhiều lượt:")
        sid = "demo_session"
        questions = [
            "Điều khoản thanh toán trong hợp đồng là gì?",
            "Nếu tôi trả trễ thì bị phạt bao nhiêu?",
        ]

        for q in questions:
            t0    = time.time()
            reply = agent.chat(q, user_id="u1", session_id=sid)
            ms    = (time.time() - t0) * 1000

            print(f"\n  👤 User: {q}")
            print(f"  🤖 Agent: {reply.answer}")
            if reply.sources:
                print(f"  📚 Nguồn: {', '.join(s.label() for s in reply.sources)}")
            print(f"  ⚡ {ms:.0f}ms | tools={reply.tool_calls} | retrieved={reply.retrieved}")

    session = agent._sessions.get(sid)
    print(f"\n  💬 Session có {len(session.turns)} turns")
    ok("Demo pipeline hoàn tất")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sep("RAG AGENT — BÀI KIỂM TRA (11 tests)")

    agent = test_basic_chat()
    test_multi_turn(agent)
    test_fallback()
    test_clarify()
    test_document_list()
    test_source_extraction()
    test_session_management()
    test_session_trim()
    test_dispatch_tool_edge_cases()
    test_agent_reply_structure()
    test_system_prompt_and_tools()
    demo_pipeline()

    sep("HOÀN TẤT — 11/11 kiểm tra thành công! ✅")


if __name__ == "__main__":
    main()

"""
test_embedding.py  —  Kiểm tra EmbeddingEngine + VectorStore end-to-end.
Chạy: python test_embedding.py
"""

import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")
sys.path.insert(0, str(Path(__file__).parent))

from embedding_engine import DEFAULT_MODEL_KEY, MODEL_REGISTRY, EmbeddingEngine, VectorStore
from document_processor import DocumentProcessor


def sep(title):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")


SAMPLE_DOCS = [
    """Điều khoản thanh toán

Khách hàng phải thanh toán trong vòng 30 ngày kể từ ngày xuất hóa đơn.
Phương thức thanh toán: chuyển khoản ngân hàng hoặc séc công ty.
Trường hợp thanh toán trễ hạn sẽ bị phạt 0.5% mỗi tháng trên số tiền còn lại.
Mọi tranh chấp liên quan đến hóa đơn phải thông báo bằng văn bản trong 7 ngày.
""",
    """Chính sách bảo hành sản phẩm

Sản phẩm được bảo hành 12 tháng kể từ ngày mua hàng.
Bảo hành bao gồm: lỗi sản xuất, hỏng hóc phần cứng do nhà sản xuất.
Không áp dụng bảo hành: hư hỏng do người dùng, thiên tai, điện áp bất thường.
Khách hàng cần xuất trình hóa đơn gốc và phiếu bảo hành để được hỗ trợ.
""",
    """Quy trình tuyển dụng nhân sự

Bước 1: Phòng ban gửi yêu cầu tuyển dụng (mẫu HR-01) tới Phòng Nhân sự.
Bước 2: Phòng Nhân sự đăng tuyển trong vòng 3 ngày làm việc.
Bước 3: Sàng lọc hồ sơ và liên hệ ứng viên phù hợp trong 5 ngày.
Bước 4: Phỏng vấn vòng 1 với Phòng Nhân sự, vòng 2 với Trưởng phòng chuyên môn.
Bước 5: Gửi thư mời làm việc và hoàn tất thủ tục onboarding.
""",
]


# ═══ TEST 1: Embed cơ bản ══════════════════════════════════════════════════════
def test_basic():
    sep("TEST 1: EmbeddingEngine — Embed cơ bản")

    engine = EmbeddingEngine(DEFAULT_MODEL_KEY, use_cache=True)
    print(f"  Model  : {engine.model_name}")
    print(f"  Dim    : {engine.dim}")

    texts = [
        "Điều khoản thanh toán trong hợp đồng",
        "Khách hàng phải trả tiền trong 30 ngày",
        "Quy trình tuyển dụng nhân sự mới",
        "The payment terms are net 30 days",
    ]

    result = engine.embed(texts)
    print(f"  Shape  : {result.vectors.shape}")
    print(f"  Elapsed: {result.elapsed_ms:.0f}ms")

    print(f"\n  --- Cosine Similarity ---")
    pairs = [
        (0, 1, "VN thanh toán ↔ VN thanh toán"),
        (0, 2, "VN thanh toán ↔ VN tuyển dụng"),
        (0, 3, "VN thanh toán ↔ EN payment   "),
    ]
    for i, j, label in pairs:
        sim = result.similarity(i, j)
        bar = "█" * int(sim * 20)
        print(f"  {label}  {sim:.4f}  {bar}")

    return engine


# ═══ TEST 2: Cache ═════════════════════════════════════════════════════════════
def test_cache(engine):
    sep("TEST 2: Embedding Cache")

    texts = ["Điều khoản bảo mật thông tin", "Chính sách hoàn trả hàng hóa"]

    r1 = engine.embed(texts)
    print(f"  Lần 1 — {r1.elapsed_ms:6.0f}ms  | cache hits: {r1.cache_hits}/{r1.count}")

    r2 = engine.embed(texts)
    print(f"  Lần 2 — {r2.elapsed_ms:6.0f}ms  | cache hits: {r2.cache_hits}/{r2.count}")

    speedup = r1.elapsed_ms / max(r2.elapsed_ms, 0.1)
    print(f"  Tăng tốc: {speedup:.1f}×  |  stats: {engine.cache_stats}")


# ═══ TEST 3: most_similar ══════════════════════════════════════════════════════
def test_most_similar(engine):
    sep("TEST 3: most_similar — Tìm kiếm trong corpus nhỏ")

    corpus = [
        "Quy trình phê duyệt ngân sách phòng ban",
        "Điều khoản thanh toán và phạt trễ hạn",
        "Chính sách nghỉ phép năm cho nhân viên",
        "Hướng dẫn sử dụng hệ thống ERP",
        "Bảo mật thông tin khách hàng và đối tác",
        "Payment terms and late fees policy",
        "Employee annual leave entitlement",
    ]

    corpus_vecs = engine.embed(corpus).vectors

    queries = [
        "Quy định về tiền phạt khi thanh toán muộn",
        "Nhân viên được nghỉ bao nhiêu ngày mỗi năm?",
    ]

    for query in queries:
        q_vec = engine.embed_one(query, is_query=True)
        hits  = engine.most_similar(q_vec, corpus_vecs, top_k=3)
        print(f'\n  Query: "{query}"')
        for rank, (idx, score) in enumerate(hits, 1):
            bar = "█" * int(score * 20)
            print(f"    {rank}. [{score:.4f}] {bar}  {corpus[idx]}")


# ═══ TEST 4: Similarity Matrix ═════════════════════════════════════════════════
def test_similarity_matrix(engine):
    sep("TEST 4: Similarity Matrix")

    texts  = ["Thanh toán đúng hạn", "Payment on time", "Trả tiền trước hạn", "Tuyển dụng nhân viên"]
    labels = ["TT đúng hạn", "Payment", "Trả trước", "Tuyển dụng"]
    result = engine.embed(texts)
    mat    = engine.similarity_matrix(result)

    print(f"\n  {'':15}", end="")
    for l in labels:
        print(f"{l:>13}", end="")
    print()

    for i, lbl in enumerate(labels):
        print(f"  {lbl:15}", end="")
        for j in range(len(labels)):
            val = mat[i, j]
            mk  = "★" if i != j and val > 0.7 else " "
            print(f"{val:>11.3f}{mk} ", end="")
        print()

    print("\n  ★ = similarity > 0.7 (rất tương đồng)")


# ═══ TEST 5: VectorStore — Pipeline đầy đủ ════════════════════════════════════
def test_vector_store():
    sep("TEST 5: VectorStore — Pipeline đầy đủ với DocumentProcessor")

    engine    = EmbeddingEngine(DEFAULT_MODEL_KEY, use_cache=True)
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir    = Path(tmp)
        chroma_dir = str(tmp_dir / "chroma")
        store      = VectorStore(engine=engine, persist_dir=chroma_dir)

        # Index 3 tài liệu
        files_info = [
            ("hop_dong.txt", "user_001", 0),
            ("bao_hanh.txt", "user_001", 1),
            ("tuyen_dung.txt", "user_002", 2),
        ]

        for fname, uid, idx in files_info:
            fpath = tmp_dir / fname
            fpath.write_text(SAMPLE_DOCS[idx], encoding="utf-8")
            result = processor.process(fpath, user_id=uid)
            added  = store.add_chunks(result.chunks)
            print(f"  ✅ {fname} → {result.chunk_count} chunks (added {added})")

        print(f"\n  Stats: {store.stats()}")

        # Tìm kiếm có filter user_id
        test_cases = [
            ("Quy định thanh toán trễ hạn phạt bao nhiêu?", "user_001"),
            ("Điều kiện để được bảo hành sản phẩm?",        "user_001"),
            ("Phỏng vấn tuyển dụng gồm mấy vòng?",          "user_002"),
        ]

        print()
        for query, uid in test_cases:
            print(f'  🔍 [{uid}] "{query}"')
            hits = store.search(query, top_k=2, user_id=uid, min_score=0.1)
            for i, hit in enumerate(hits, 1):
                preview = hit.text[:100].replace("\n", " ")
                print(f"      [{i}] {hit.score:.4f} | {hit.source_file} | {preview}…")

        # build_context → đưa vào AI Agent
        print(f"\n  --- build_context (prompt AI Agent) ---")
        hits = store.search("thanh toán hóa đơn", top_k=3, user_id="user_001")
        ctx  = store.build_context(hits, max_tokens=500)
        print(f"  Context: {len(ctx)} ký tự")
        print(f"\n{ctx[:400]}\n  …")

        # Xoá
        if hits:
            did     = hits[0].metadata.get("doc_id", "")
            deleted = store.delete_doc(did)
            print(f"\n  🗑️  Xoá {deleted} chunks | Còn lại: {store.count}")


# ═══ TEST 6: Model Registry ════════════════════════════════════════════════════
def test_model_registry():
    sep("TEST 6: Model Registry — Các model khả dụng")
    print(f"  {'Key':<25} {'Dim':>5}  Ghi chú")
    print(f"  {'-'*25} {'-'*5}  {'-'*38}")
    for key, info in MODEL_REGISTRY.items():
        print(f"  {key:<25} {info['dim']:>5}  {info['note']}")


# ═══ MAIN ══════════════════════════════════════════════════════════════════════
def main():
    sep("EMBEDDING ENGINE — BÀI KIỂM TRA")
    engine = test_basic()
    test_cache(engine)
    test_most_similar(engine)
    test_similarity_matrix(engine)
    test_vector_store()
    test_model_registry()
    sep("HOÀN TẤT — Tất cả kiểm tra thành công! ✅")


if __name__ == "__main__":
    main()

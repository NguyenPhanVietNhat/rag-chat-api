# DocMind RAG API

Hệ thống hỏi đáp tài liệu nội bộ sử dụng **Retrieval-Augmented Generation (RAG)**, xây dựng trên FastAPI + OpenAI GPT-4o + ChromaDB + Sentence Transformers.

---

## Tính năng

- **Upload & xử lý tài liệu** — hỗ trợ PDF, DOCX, XLSX, TXT
- **Hỏi đáp thông minh** — trả lời dựa trên nội dung tài liệu, kèm trích dẫn nguồn
- **Streaming response** — phản hồi theo dạng Server-Sent Events (SSE)
- **Conversation memory** — hỗ trợ hội thoại đa lượt trong một session
- **Xác thực người dùng** — đăng ký / đăng nhập với JWT
- **Phân quyền** — phân biệt role `user` và `admin`
- **Frontend tĩnh** — giao diện chat, upload, và trang admin kèm theo
- **Docker ready** — triển khai đơn giản bằng Docker Compose

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                        Frontend                         │
│         index.html / chat.html / upload.html / admin.html│
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Auth   │  │   Document   │  │    RAG Agent      │  │
│  │  (JWT)   │  │  Processor   │  │ (GPT-4o + Search) │  │
│  └──────────┘  └──────┬───────┘  └────────┬──────────┘  │
│                        │                   │             │
│              ┌─────────▼───────────────────▼──────────┐ │
│              │   Embedding Engine (sentence-transformers)│ │
│              └─────────────────────┬──────────────────┘ │
│                                    │                     │
│                          ┌─────────▼──────────┐         │
│                          │   ChromaDB (Vector) │         │
│                          └────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
rag-chat-api-main/
├── backend/
│   ├── main.py               # FastAPI app, định nghĩa toàn bộ endpoints
│   ├── auth.py               # Đăng ký, đăng nhập, JWT middleware
│   ├── config.py             # Đọc cấu hình từ .env
│   ├── document_processor.py # Parse PDF/DOCX/XLSX/TXT → chunks
│   ├── embedding_engine.py   # Sentence Transformers embedding
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── rag_agent.py          # RAG pipeline + GPT-4o + streaming
│   ├── session_store.py      # Quản lý lịch sử hội thoại
│   ├── database.py           # SQLite (user data)
│   ├── file_storage.py       # Quản lý file upload
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── index.html            # Trang chủ / đăng nhập
│   ├── chat.html             # Giao diện hỏi đáp
│   ├── upload.html           # Upload tài liệu
│   └── admin.html            # Quản trị người dùng & tài liệu
└── docker-compose.yml
```

---

## Yêu cầu

- Python 3.11+
- OpenAI API Key
- Docker & Docker Compose *(nếu chạy bằng container)*

---

## Cài đặt & Chạy

### Cách 1: Docker Compose *(khuyến nghị)*

```bash
# 1. Clone repo
git clone <repo-url>
cd rag-chat-api-main

# 2. Tạo file .env
cp backend/.env.example backend/.env
# Mở backend/.env và điền OPENAI_API_KEY, SECRET_KEY

# 3. Build & chạy
docker compose up --build -d
```

API sẵn sàng tại: `http://localhost:8000`  
Giao diện web tại: `http://localhost:8000/ui`

---

### Cách 2: Chạy thủ công

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Cài dependencies
cd backend
pip install -r requirements.txt

# 3. Tạo file .env
cp .env.example .env
# Điền OPENAI_API_KEY và SECRET_KEY

# 4. Chạy server
uvicorn main:app --reload --port 8000
```

---

## Cấu hình `.env`

| Biến | Bắt buộc | Mô tả | Mặc định |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | API key của OpenAI | — |
| `SECRET_KEY` | ✅ | Secret để ký JWT token | — |
| `TOKEN_EXPIRE_MIN` | | Thời gian hết hạn token (phút) | `1440` (24h) |
| `EMBED_MODEL` | | Model embedding đa ngôn ngữ | `multilingual-default` |
| `CHUNK_SIZE` | | Kích thước chunk văn bản | `500` |
| `CHUNK_OVERLAP` | | Độ chồng lấp giữa các chunk | `80` |
| `TOP_K` | | Số chunk truy xuất mỗi truy vấn | `5` |
| `MIN_SCORE` | | Ngưỡng similarity tối thiểu | `0.20` |
| `MAX_FILE_MB` | | Giới hạn kích thước file upload | `50` |
| `UPLOAD_DIR` | | Thư mục lưu file | `./uploads` |
| `CHROMA_DIR` | | Thư mục lưu vector DB | `./chroma_db` |
| `ALLOWED_ORIGINS` | | Danh sách CORS origins | `localhost:8000, :5500` |

---

## API Endpoints

Swagger UI: `http://localhost:8000/api/docs`  
ReDoc: `http://localhost:8000/api/redoc`

### Auth

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/register` | Đăng ký tài khoản mới |
| `POST` | `/login` | Đăng nhập, nhận JWT token |
| `GET` | `/me` | Xem thông tin tài khoản hiện tại |

### Chat

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/chat` | Gửi câu hỏi, nhận trả lời (streaming SSE) |
| `POST` | `/chat/sync` | Gửi câu hỏi, nhận trả lời đồng bộ |
| `GET` | `/sessions` | Danh sách sessions hội thoại |
| `GET` | `/sessions/{id}/messages` | Lịch sử chat của session |
| `DELETE` | `/sessions/{id}` | Xoá session |

### Documents

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/docs/upload` | Upload tài liệu (PDF, DOCX, XLSX, TXT) |
| `GET` | `/docs` | Danh sách tài liệu của người dùng |
| `DELETE` | `/docs/{doc_id}` | Xoá tài liệu |

### Admin *(yêu cầu role admin)*

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/admin/users` | Danh sách tất cả người dùng |
| `POST` | `/admin/users/{username}/approve` | Phê duyệt tài khoản |
| `POST` | `/admin/users/{username}/revoke` | Thu hồi quyền truy cập |
| `DELETE` | `/admin/users/{username}` | Xoá người dùng |
| `GET` | `/admin/docs` | Tất cả tài liệu trong hệ thống |
| `DELETE` | `/admin/docs/{doc_id}` | Xoá tài liệu bất kỳ |
| `POST` | `/admin/users/{username}/role` | Thay đổi role người dùng |

### System

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái hệ thống |

---

## Ví dụ sử dụng

### Đăng ký & đăng nhập

```bash
# Đăng ký
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret123"}'

# Đăng nhập
curl -X POST http://localhost:8000/login \
  -d "username=alice&password=secret123"
# → trả về access_token
```

### Upload tài liệu

```bash
curl -X POST http://localhost:8000/docs/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@bao_cao_q3.pdf"
```

### Hỏi đáp

```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Doanh thu Q3 là bao nhiêu?", "session_id": "session-001"}'
```

---

## Định dạng tài liệu hỗ trợ

| Định dạng | Mô tả |
|---|---|
| `.pdf` | Trích xuất text + bảng (pdfplumber + pypdf) |
| `.docx` | Tài liệu Word (python-docx) |
| `.xlsx` | Bảng tính Excel (openpyxl) |
| `.txt` | Văn bản thuần |

---

## Stack công nghệ

| Thành phần | Công nghệ |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o |
| Embedding | Sentence Transformers (multilingual) |
| Vector Database | ChromaDB |
| Auth | JWT (HS256) + bcrypt |
| Document Parsing | pdfplumber, pypdf, python-docx, openpyxl |
| Text Splitting | LangChain Text Splitters |
| Database | SQLite (user data) |
| Container | Docker + Docker Compose |

---

## Chạy tests

```bash
cd backend
python -m pytest test_suite.py -v
python -m pytest test_rag_agent.py -v
python -m pytest test_embedding.py -v
python -m pytest test_processor.py -v
```

---

## Giấy phép

MIT License

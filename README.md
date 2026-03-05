# 🤖 AI Customer Support Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers customer support questions by searching your uploaded documents and generating accurate answers using an LLM.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Document upload** | PDF, TXT, Markdown |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key needed) |
| **Vector store** | ChromaDB (persistent) |
| **LLM backend** | Local rule-based (default) or OpenAI GPT |
| **API** | FastAPI with Swagger UI |
| **UI** | Streamlit with premium dark theme |

---

## 📁 Project Structure

```
ai-support-assistant/
├── app.py                  # FastAPI backend (POST /upload, POST /ask)
├── streamlit_app.py        # Streamlit frontend UI
├── rag_pipeline.py         # Core RAG pipeline
├── vector_store.py         # ChromaDB wrapper
├── embedding_model.py      # Embedding model (HuggingFace / OpenAI)
├── upload_docs.py          # Document loader & text splitter
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── sample_docs/            # Sample support documents for testing
│   ├── refund_policy.txt
│   ├── faq.md
│   └── product_manual.txt
└── chroma_db/              # Auto-created: persistent vector store
```

---

## ⚙️ Setup

### 1. Create & activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

Edit `.env` — for local usage (no API key needed), keep the defaults:
```env
EMBEDDING_BACKEND=huggingface
LLM_BACKEND=local
```

To use OpenAI GPT for better answers:
```env
EMBEDDING_BACKEND=openai
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
```

---

## 🏃 Running the Application

### Start the FastAPI backend
```bash
uvicorn app:app --reload --port 8000
```
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/status

### Start the Streamlit UI (new terminal)
```bash
streamlit run streamlit_app.py
```
- Opens at: http://localhost:8501

---

## 🔌 API Reference

### `POST /upload`
Upload a support document.
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_docs/refund_policy.txt"
```
Response:
```json
{
  "filename": "refund_policy.txt",
  "chunks_added": 12,
  "total_chunks": 12,
  "message": "'refund_policy.txt' processed and indexed successfully."
}
```

### `POST /ask`
Ask a question.
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```
Response:
```json
{
  "answer": "Refunds are allowed within 7 days of the purchase date.",
  "source": "refund_policy.txt",
  "confidence": 0.88
}
```

### `GET /status`
```bash
curl http://localhost:8000/status
```

### `DELETE /reset`
Clear all indexed documents.
```bash
curl -X DELETE http://localhost:8000/reset
```

---

## 🧪 Quick Test

```bash
# 1. Start the backend
uvicorn app:app --reload

# 2. Upload sample documents
curl -X POST http://localhost:8000/upload -F "file=@sample_docs/refund_policy.txt"
curl -X POST http://localhost:8000/upload -F "file=@sample_docs/faq.md"

# 3. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do I have to return a product?"}'
```

---

## 🏗️ Architecture

```
User Question
     ↓
[Embedding Model]          ← sentence-transformers (local)
     ↓
[ChromaDB Vector Store]    ← semantic similarity search
     ↓
[Top-K Relevant Chunks]
     ↓
[LLM (OpenAI / Local)]     ← answer generation with context
     ↓
Final Answer + Source + Confidence
```

---

## 📦 Tech Stack

- **FastAPI** — REST API framework
- **ChromaDB** — vector database
- **LangChain** — RAG orchestration
- **sentence-transformers** — local HuggingFace embeddings
- **Streamlit** — frontend UI
- **pypdf** — PDF text extraction

---

## 🔒 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_BACKEND` | `huggingface` | `huggingface` or `openai` |
| `HUGGINGFACE_MODEL` | `all-MiniLM-L6-v2` | HF model name |
| `LLM_BACKEND` | `local` | `local` or `openai` |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `500` | Text chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

---

## 💡 Extending the System

- **Add conversation memory**: Use `ConversationBufferMemory` from LangChain
- **Stream responses**: Use FastAPI `StreamingResponse` + OpenAI streaming
- **Add authentication**: Use FastAPI's `Depends` with JWT tokens
- **Switch vector DB**: Replace `VectorStore` with Pinecone or Weaviate
- **Deploy**: Dockerise the app or deploy to Railway / Render
"# ai-support-assistant" 

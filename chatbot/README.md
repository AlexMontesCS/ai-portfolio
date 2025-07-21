# Conversational AI Chatbot with RAG

A production-ready conversational AI chatbot using Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents.

## Features

- **Document Processing**: Support for PDF and Markdown with intelligent text chunking.
- **Advanced RAG**: Implements a sophisticated Retrieval-Augmented Generation pipeline, including hybrid search, query reformulation, and entity recognition.
- **Verifiable Citations**: Provides automatic source tracking to link responses back to the original documents.
- **Modern User Interface**: A responsive, real-time chat interface built with React.
## Quick Start

### 1. Setup Environment

**Backend Setup:**
```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

**Frontend Setup:**
```bash
cd frontend
npm install
```

### 2. Start the Application

**Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm start
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## API Endpoints

- `POST /api/documents/upload` - Upload documents
- `GET /api/documents/` - List documents
- `DELETE /api/documents/{doc_id}` - Delete document
- `POST /api/chat/` - Chat with RAG system
- `GET /health` - System health status

## Environment Configuration

Key variables for `backend/.env`:
```
OPENAI_API_KEY=your_key_here
CHAT_MODEL=gpt-4-turbo
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_STORE_TYPE=faiss
```

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, LangChain, FAISS, sentence-transformers
- **Frontend**: React 18, TypeScript, React Query, Tailwind CSS
- **Models**: Compatible with OpenAI and Google Gemini API

## Troubleshooting

- For API key errors, check `backend/.env` configuration
- For CORS issues, ensure frontend runs on port 3000
- See logs in `backend/chatbot.log` for detailed error information
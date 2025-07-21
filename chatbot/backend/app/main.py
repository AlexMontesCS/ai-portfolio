from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

from .core.config import settings
from .api import documents, chat
from .models.schemas import HealthResponse
from .services.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chatbot.log") if Path("chatbot.log").parent.exists() else logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting up RAG Chatbot application...")
    
    try:
        # Initialize vector store
        vector_store = VectorStoreManager()
        health = vector_store.health_check()
        logger.info(f"Vector store initialized: {health}")
        
        # Store in app state
        app.state.vector_store = vector_store
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot application...")


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A conversational AI chatbot with Retrieval-Augmented Generation capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Include routers
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    try:
        # Check vector store health using the shared instance
        vector_store = request.app.state.vector_store
        vector_health = vector_store.health_check()
        
        # Get actual document count from document service
        from .services.document_service import DocumentService
        doc_service = DocumentService(vector_store=vector_store)
        documents = doc_service.get_documents()
        actual_document_count = len(documents)
        
        # Get chunks count from document metadata (more accurate)
        total_chunks = sum(doc.chunk_count for doc in documents)
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_status=vector_health.get("status", "unknown"),
            documents_count=actual_document_count,
            chunks_count=total_chunks
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            vector_store_status="error",
            documents_count=0,
            chunks_count=0
        )


@app.get("/debug/vector-store")
async def debug_vector_store(request: Request):
    """Debug endpoint to check vector store state."""
    try:
        vector_store = request.app.state.vector_store
        
        # Try a simple search
        test_results = vector_store.search("Alex", k=1)
        
        return {
            "vector_store_type": type(vector_store).__name__,
            "store_type": type(vector_store.store).__name__ if hasattr(vector_store, 'store') else "N/A",
            "index_path": vector_store.store.index_path if hasattr(vector_store, 'store') and hasattr(vector_store.store, 'index_path') else "N/A",
            "vectorstore_exists": vector_store.store.vectorstore is not None if hasattr(vector_store, 'store') and hasattr(vector_store.store, 'vectorstore') else False,
            "document_metadata_count": len(vector_store.store.document_metadata) if hasattr(vector_store, 'store') and hasattr(vector_store.store, 'document_metadata') else 0,
            "test_search_results": len(test_results),
            "test_search_content": [doc.page_content[:100] for doc in test_results] if test_results else [],
            "vectorstore_docstore_count": len(vector_store.store.vectorstore.docstore._dict) if hasattr(vector_store, 'store') and vector_store.store.vectorstore and hasattr(vector_store.store.vectorstore, 'docstore') else 0
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/document-content/{document_id}")
async def debug_document_content(document_id: str, request: Request):
    """Debug endpoint to see the full extracted text content of a document."""
    try:
        from .services.document_service import DocumentService
        from pathlib import Path
        from .models.schemas import DocumentType
        
        vector_store = request.app.state.vector_store
        document_service = DocumentService(vector_store=vector_store)
        
        # Get document metadata
        doc_metadata = document_service.documents_metadata.get(document_id)
        if not doc_metadata:
            return {"error": f"Document {document_id} not found"}
        
        # Read the file and extract text
        file_path = Path(doc_metadata["file_path"])
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Extract text based on document type
        doc_type = DocumentType(doc_metadata["document_type"])
        if doc_type == DocumentType.PDF:
            text = document_service._extract_text_from_pdf(str(file_path))
        elif doc_type == DocumentType.MARKDOWN:
            text = document_service._extract_text_from_markdown(str(file_path))
        else:
            text = document_service._extract_text_from_text(str(file_path))
        
        return {
            "document_id": document_id,
            "filename": doc_metadata["filename"],
            "document_type": doc_metadata["document_type"],
            "text_length": len(text),
            "full_text": text,
            "contains_teaching_lab_studio": "teaching lab studio" in text.lower(),
            "contains_lab_studio": "lab studio" in text.lower(),
            "contains_teaching": "teaching" in text.lower()
        }
    except Exception as e:
        return {"error": str(e)}
async def reinitialize_vector_store(request: Request):
    """Debug endpoint to reinitialize the vector store with current documents."""
    try:
        from .services.document_service import DocumentService
        
        # Get a fresh document service instance
        vector_store = request.app.state.vector_store
        document_service = DocumentService(vector_store=vector_store)
        
        # Force re-indexing of all documents
        document_service._reindex_all_documents()
        
        # Get updated counts
        documents = document_service.get_documents()
        documents_count = len(documents)
        chunks_count = sum(doc.chunk_count for doc in documents)
        
        return {
            "message": "Vector store reinitialized successfully",
            "documents_count": documents_count,
            "chunks_count": chunks_count
        }
    except Exception as e:
        return {"error": str(e)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use datetime.utcnow() in production
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "status_code": 500,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )


@app.get("/debug/document-chunks/{document_id}")
async def debug_document_chunks(document_id: str, request: Request):
    """Debug endpoint to show all chunks for a document."""
    try:
        from .services.document_service import DocumentService
        
        vector_store = request.app.state.vector_store
        doc_service = DocumentService(vector_store=vector_store)
        
        # Get document metadata
        doc_metadata = doc_service.documents_metadata.get(document_id)
        if not doc_metadata:
            return {"error": "Document not found"}
        
        # Get all chunks for this document from vector store
        all_results = vector_store.search("the", k=100, with_scores=True)  # Get many results with scores
        
        # Filter chunks for this document
        doc_chunks = []
        for doc, score in all_results:
            if doc.metadata.get("document_id") == document_id:
                doc_chunks.append({
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content": doc.page_content,
                    "content_length": len(doc.page_content),
                    "contains_teaching_lab_studio": "Teaching Lab Studio" in doc.page_content,
                    "contains_teaching": "teaching" in doc.page_content.lower(),
                    "contains_lab": "lab" in doc.page_content.lower(),
                })
        
        # Sort by chunk index
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        
        return {
            "document_id": document_id,
            "filename": doc_metadata.get("filename"),
            "total_chunks": len(doc_chunks),
            "chunks": doc_chunks
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

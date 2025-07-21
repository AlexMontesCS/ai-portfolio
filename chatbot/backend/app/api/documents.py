from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import aiofiles
from ..models.schemas import (
    DocumentResponse, DocumentListResponse, ErrorResponse,
    VectorSearchRequest, VectorSearchResponse
)
from ..services.document_service import DocumentService
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Dependency to get document service
def get_document_service(request: Request) -> DocumentService:
    try:
        # Get the vector store from app state instead of creating a new one
        vector_store = request.app.state.vector_store
        return DocumentService(vector_store=vector_store)
    except Exception as e:
        logger.error(f"Failed to create DocumentService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is currently unavailable"
        )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service)
):
    """Upload and process a document (PDF, Markdown, or text)."""
    try:
        # Validate file size
        if file.size and file.size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
            )
        
        # Validate file type
        allowed_types = {
            'application/pdf',
            'text/markdown',
            'text/plain',
            'text/x-markdown',
            'application/octet-stream'  # Sometimes markdown files come as this
        }
        
        if file.content_type not in allowed_types and not any(
            file.filename.lower().endswith(ext) for ext in ['.pdf', '.md', '.markdown', '.txt']
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Please upload PDF, Markdown, or text files."
            )
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Process the document
        document_response = await document_service.upload_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        
        logger.info(f"Successfully uploaded document: {file.filename}")
        return document_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def get_documents(
    document_service: DocumentService = Depends(get_document_service)
):
    """Get list of all uploaded documents."""
    try:
        documents = document_service.get_documents()
        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get specific document by ID."""
    try:
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Delete a document and remove it from the vector store."""
    try:
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or could not be deleted"
            )
        
        return {"message": "Document deleted successfully", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.post("/search", response_model=VectorSearchResponse)
async def search_documents(
    request: VectorSearchRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """Search for relevant document chunks using vector similarity."""
    try:
        logger.info(f"Search API called with query: '{request.query}', top_k: {request.top_k}")
        
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        logger.info("Calling document_service.search_documents...")
        search_results = document_service.search_documents(
            query=request.query,
            top_k=request.top_k
        )
        logger.info(f"Got {len(search_results)} raw search results")
        
        # Filter by threshold
        filtered_results = [
            result for result in search_results
            if result.get("relevance_score", 0) >= request.threshold
        ]
        logger.info(f"After threshold filtering: {len(filtered_results)} results")
        
        # Convert to response format
        vector_results = []
        for result in filtered_results:
            vector_results.append({
                "chunk_id": result.get("chunk_id", ""),
                "document_id": result.get("document_id", ""),
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("relevance_score", 0.0)
            })
        
        return VectorSearchResponse(
            results=vector_results,
            query=request.query,
            total_results=len(vector_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search documents"
        )


@router.get("/{document_id}/summary")
async def get_document_summary(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Generate a summary of the specified document."""
    try:
        # Check if document exists
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Note: In a full implementation, you might want to use the RAG service here
        # For now, we'll return a placeholder
        return {
            "document_id": document_id,
            "filename": document.filename,
            "summary": "Document summary functionality would be implemented here using the RAG service."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate document summary"
        )


@router.get("/stats/overview")
async def get_document_stats(
    document_service: DocumentService = Depends(get_document_service)
):
    """Get document collection statistics."""
    try:
        stats = document_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document statistics"
        )

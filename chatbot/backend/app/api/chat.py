from fastapi import APIRouter, HTTPException, status, Depends, Request
from typing import List, Optional
from ..models.schemas import (
    ChatRequest, ChatResponse, ConversationResponse, ChatMessage
)
from ..services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Dependency to get RAG service
def get_rag_service(request: Request) -> RAGService:
    try:
        # Get the vector store from app state instead of creating a new one
        vector_store = request.app.state.vector_store
        return RAGService(vector_store=vector_store)
    except Exception as e:
        logger.error(f"Failed to create RAGService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is currently unavailable"
        )


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Send a message to the RAG chatbot."""
    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        response = await rag_service.chat(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat message: {str(e)}"
        )


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Create a new conversation."""
    try:
        conversation = rag_service.create_conversation()
        return conversation
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get all conversation histories."""
    try:
        conversations = rag_service.get_conversations()
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get a specific conversation by ID."""
    try:
        conversation = rag_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Delete a conversation."""
    try:
        success = rag_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": "Conversation deleted successfully", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )


@router.post("/summarize/{document_id}")
async def summarize_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Generate a summary of a specific document using the RAG system."""
    try:
        summary = rag_service.summarize_document(document_id)
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or could not be summarized"
            )
        
        return {
            "document_id": document_id,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate document summary"
        )

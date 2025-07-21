from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DocumentUploadRequest(BaseModel):
    filename: str
    content_type: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    document_type: DocumentType
    upload_time: datetime
    size: int
    chunk_count: int
    status: str = "processed"


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    relevance_score: float
    text_snippet: str = Field(..., description="Relevant text snippet from the document")


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: Optional[List[Citation]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    include_citations: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    message: ChatMessage
    conversation_id: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_status: str
    documents_count: int
    chunks_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.3


class VectorSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class VectorSearchResponse(BaseModel):
    results: List[VectorSearchResult]
    query: str
    total_results: int

export interface Document {
  id: string;
  filename: string;
  document_type: 'pdf' | 'markdown' | 'text';
  upload_time: string;
  size: number;
  chunk_count: number;
  status: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface Citation {
  document_id: string;
  document_name: string;
  page_number?: number;
  chunk_id: string;
  relevance_score: number;
  text_snippet: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  citations?: Citation[];
  metadata?: Record<string, any>;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  include_citations?: boolean;
  max_tokens?: number;
  temperature?: number;
}

export interface ChatResponse {
  message: ChatMessage;
  conversation_id: string;
  tokens_used?: number;
  processing_time?: number;
}

export interface Conversation {
  conversation_id: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export interface VectorSearchRequest {
  query: string;
  top_k?: number;
  threshold?: number;
}

export interface VectorSearchResult {
  chunk_id: string;
  document_id: string;
  content: string;
  metadata: Record<string, any>;
  score: number;
}

export interface VectorSearchResponse {
  results: VectorSearchResult[];
  query: string;
  total_results: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  vector_store_status: string;
  documents_count: number;
  chunks_count: number;
}

export interface DocumentStats {
  total_documents: number;
  total_chunks: number;
  total_size_bytes: number;
  document_types: Record<string, number>;
  vector_store_count: number;
}

export interface ApiError {
  error: string;
  detail?: string;
  timestamp: string;
  status_code?: number;
}

import axios from 'axios';
import {
  Document,
  DocumentListResponse,
  ChatRequest,
  ChatResponse,
  Conversation,
  VectorSearchRequest,
  VectorSearchResponse,
  HealthResponse,
  DocumentStats,
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const documentsApi = {
  // Upload a document
  uploadDocument: async (file: File): Promise<Document> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get all documents
  getDocuments: async (): Promise<DocumentListResponse> => {
    const response = await api.get('/api/documents/');
    return response.data;
  },

  // Get specific document
  getDocument: async (documentId: string): Promise<Document> => {
    const response = await api.get(`/api/documents/${documentId}`);
    return response.data;
  },

  // Delete document
  deleteDocument: async (documentId: string): Promise<void> => {
    await api.delete(`/api/documents/${documentId}`);
  },

  // Search documents
  searchDocuments: async (request: VectorSearchRequest): Promise<VectorSearchResponse> => {
    const response = await api.post('/api/documents/search', request);
    return response.data;
  },

  // Get document summary
  getDocumentSummary: async (documentId: string): Promise<{ document_id: string; summary: string }> => {
    const response = await api.get(`/api/documents/${documentId}/summary`);
    return response.data;
  },

  // Get document stats
  getDocumentStats: async (): Promise<DocumentStats> => {
    const response = await api.get('/api/documents/stats/overview');
    return response.data;
  },
};

export const chatApi = {
  // Create new conversation
  createConversation: async (): Promise<Conversation> => {
    const response = await api.post('/api/chat/conversations');
    return response.data;
  },

  // Send chat message
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post('/api/chat/', request);
    return response.data;
  },

  // Get all conversations
  getConversations: async (): Promise<Conversation[]> => {
    const response = await api.get('/api/chat/conversations');
    return response.data;
  },

  // Get specific conversation
  getConversation: async (conversationId: string): Promise<Conversation> => {
    const response = await api.get(`/api/chat/conversations/${conversationId}`);
    return response.data;
  },

  // Delete conversation
  deleteConversation: async (conversationId: string): Promise<void> => {
    await api.delete(`/api/chat/conversations/${conversationId}`);
  },

  // Summarize document
  summarizeDocument: async (documentId: string): Promise<{ document_id: string; summary: string }> => {
    const response = await api.post(`/api/chat/summarize/${documentId}`);
    return response.data;
  },
};

export const healthApi = {
  // Get health status
  getHealth: async (): Promise<HealthResponse> => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;

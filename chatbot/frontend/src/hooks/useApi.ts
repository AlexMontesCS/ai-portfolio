import { useQuery, useMutation, useQueryClient } from 'react-query';
import { documentsApi, chatApi, healthApi } from '../services/api';
import {
  Document,
  ChatRequest,
  ChatResponse,
  Conversation,
  VectorSearchRequest,
} from '../types';

// Document hooks
export const useDocuments = () => {
  return useQuery('documents', documentsApi.getDocuments, {
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useDocument = (documentId: string) => {
  return useQuery(
    ['document', documentId],
    () => documentsApi.getDocument(documentId),
    {
      enabled: !!documentId,
    }
  );
};

export const useUploadDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation(documentsApi.uploadDocument, {
    onSuccess: () => {
      queryClient.invalidateQueries('documents');
      queryClient.invalidateQueries('documentStats');
      queryClient.invalidateQueries('health'); // Refresh health status immediately
    },
    onError: (error) => {
      console.error('Upload error:', error);
    },
  });
};

export const useDeleteDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation(documentsApi.deleteDocument, {
    onSuccess: () => {
      queryClient.invalidateQueries('documents');
      queryClient.invalidateQueries('documentStats');
      queryClient.invalidateQueries('health'); // Refresh health status immediately
    },
  });
};

export const useSearchDocuments = () => {
  return useMutation(documentsApi.searchDocuments);
};

export const useDocumentStats = () => {
  return useQuery('documentStats', documentsApi.getDocumentStats, {
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

// Chat hooks
export const useCreateConversation = () => {
  const queryClient = useQueryClient();
  
  return useMutation(chatApi.createConversation, {
    onSuccess: (newConversation) => {
      queryClient.invalidateQueries('conversations');
      // Pre-populate the cache for the new conversation
      queryClient.setQueryData(['conversation', newConversation.conversation_id], newConversation);
    },
  });
};

export const useSendMessage = () => {
  const queryClient = useQueryClient();
  
  return useMutation(
    (request: ChatRequest) => chatApi.sendMessage(request),
    {
      onSuccess: (data) => {
        // Invalidate conversations to refresh the list
        queryClient.invalidateQueries('conversations');
        
        // Update specific conversation if it exists
        if (data.conversation_id) {
          queryClient.invalidateQueries(['conversation', data.conversation_id]);
        }
      },
    }
  );
};

export const useConversations = () => {
  return useQuery('conversations', chatApi.getConversations, {
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useConversation = (conversationId: string) => {
  return useQuery(
    ['conversation', conversationId],
    () => chatApi.getConversation(conversationId),
    {
      enabled: !!conversationId && conversationId.length > 0,
      staleTime: 30 * 1000, // 30 seconds
      retry: false, // Don't retry failed requests for conversations
    }
  );
};

export const useDeleteConversation = () => {
  const queryClient = useQueryClient();
  
  return useMutation(chatApi.deleteConversation, {
    onSuccess: () => {
      queryClient.invalidateQueries('conversations');
    },
  });
};

export const useSummarizeDocument = () => {
  return useMutation(chatApi.summarizeDocument);
};

// Health hook
export const useHealth = () => {
  return useQuery('health', healthApi.getHealth, {
    staleTime: 10 * 1000, // 10 seconds - shorter stale time for more frequent updates
    refetchInterval: 30 * 1000, // Refetch every 30 seconds instead of 60
    refetchOnWindowFocus: true, // Refetch when user focuses the window
  });
};

// Custom hook for managing current conversation
export const useCurrentConversation = (conversationId: string | null) => {
  const conversationQuery = useConversation(conversationId || '');
  const sendMessageMutation = useSendMessage();
  
  const sendMessage = async (message: string) => {
    const request: ChatRequest = {
      message,
      conversation_id: conversationId || undefined,
      include_citations: true,
    };
    
    return sendMessageMutation.mutateAsync(request);
  };
  
  return {
    conversation: conversationQuery.data,
    messages: conversationQuery.data?.messages || [],
    isLoading: conversationId ? conversationQuery.isLoading : false,
    error: conversationId ? (conversationQuery.error as Error | null) : null,
    sendMessage,
    isSending: sendMessageMutation.isLoading,
    sendError: sendMessageMutation.error as Error | null,
  };
};

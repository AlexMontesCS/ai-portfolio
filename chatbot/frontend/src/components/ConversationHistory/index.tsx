import React from 'react';
import { MessageSquare, Trash2, Plus, Clock } from 'lucide-react';
import { format } from 'date-fns';
import { Button, LoadingSpinner } from '../ui';
import { useConversations, useDeleteConversation, useCreateConversation } from '../../hooks/useApi';
import { Conversation } from '../../types';

interface ConversationHistoryProps {
  currentConversationId: string | null;
  onConversationSelect: (conversationId: string) => void;
  onNewConversation: () => void;
}

export const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  currentConversationId,
  onConversationSelect,
  onNewConversation,
}) => {
  const { data: conversations, isLoading, error } = useConversations();
  const deleteConversationMutation = useDeleteConversation();
  const createConversationMutation = useCreateConversation();

  const handleDeleteConversation = async (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent selecting the conversation
    
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      try {
        await deleteConversationMutation.mutateAsync(conversationId);
        
        // If the deleted conversation was the current one, clear it
        if (conversationId === currentConversationId) {
          onNewConversation();
        }
      } catch (error) {
        console.error('Failed to delete conversation:', error);
      }
    }
  };

  const handleNewConversation = () => {
    createConversationMutation.mutate(undefined, {
      onSuccess: (newConversation) => {
        onConversationSelect(newConversation.conversation_id);
      },
      onError: (error) => {
        console.error('Failed to create conversation:', error);
      },
    });
  };

  const getConversationPreview = (conversation: Conversation): string => {
    if (!conversation.messages || conversation.messages.length === 0) {
      return 'New conversation';
    }
    
    const firstUserMessage = conversation.messages.find(msg => msg.role === 'user');
    if (firstUserMessage) {
      return firstUserMessage.content.length > 50 
        ? firstUserMessage.content.substring(0, 50) + '...'
        : firstUserMessage.content;
    }
    
    return 'New conversation';
  };

  if (isLoading) {
    return (
      <div className="p-4 flex items-center justify-center">
        <LoadingSpinner size="sm" />
        <span className="ml-2 text-sm text-gray-500">Loading conversations...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-red-800 text-sm">
            Failed to load conversations
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <MessageSquare className="w-5 h-5 mr-2" />
            Conversations
          </h3>
        </div>
        
        <Button
          onClick={handleNewConversation}
          disabled={createConversationMutation.isLoading}
          className="w-full flex items-center justify-center space-x-2"
          size="sm"
        >
          <Plus className="w-4 h-4" />
          <span>New Conversation</span>
        </Button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {!conversations || conversations.length === 0 ? (
          <div className="p-4 text-center">
            <MessageSquare className="mx-auto h-8 w-8 text-gray-400 mb-2" />
            <p className="text-sm text-gray-500">No conversations yet</p>
            <p className="text-xs text-gray-400 mt-1">
              Start a new conversation to get started
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {conversations.map((conversation) => (
              <ConversationItem
                key={conversation.conversation_id}
                conversation={conversation}
                isSelected={conversation.conversation_id === currentConversationId}
                onSelect={() => onConversationSelect(conversation.conversation_id)}
                onDelete={(e) => handleDeleteConversation(conversation.conversation_id, e)}
                isDeleting={deleteConversationMutation.isLoading}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

interface ConversationItemProps {
  conversation: Conversation;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
  isDeleting: boolean;
}

const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  isSelected,
  onSelect,
  onDelete,
  isDeleting,
}) => {
  const getConversationPreview = (conversation: Conversation): string => {
    if (!conversation.messages || conversation.messages.length === 0) {
      return 'New conversation';
    }
    
    const firstUserMessage = conversation.messages.find(msg => msg.role === 'user');
    if (firstUserMessage) {
      return firstUserMessage.content.length > 40 
        ? firstUserMessage.content.substring(0, 40) + '...'
        : firstUserMessage.content;
    }
    
    return 'New conversation';
  };

  const messageCount = conversation.messages?.length || 0;
  const lastMessageTime = conversation.messages && conversation.messages.length > 0
    ? new Date(conversation.messages[conversation.messages.length - 1].timestamp)
    : new Date(conversation.created_at);

  return (
    <div
      onClick={onSelect}
      className={`
        relative group cursor-pointer rounded-lg p-3 transition-colors
        ${isSelected 
          ? 'bg-blue-100 border border-blue-200' 
          : 'hover:bg-gray-50 border border-transparent'
        }
      `}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-1 mb-1">
            <MessageSquare className="w-3 h-3 text-gray-400 flex-shrink-0" />
            <span className="text-xs text-gray-500">
              {messageCount} message{messageCount !== 1 ? 's' : ''}
            </span>
          </div>
          
          <p className={`text-sm font-medium truncate ${
            isSelected ? 'text-blue-900' : 'text-gray-900'
          }`}>
            {getConversationPreview(conversation)}
          </p>
          
          <div className="flex items-center space-x-1 mt-1">
            <Clock className="w-3 h-3 text-gray-400" />
            <span className="text-xs text-gray-500">
              {format(lastMessageTime, 'MMM d, h:mm a')}
            </span>
          </div>
        </div>
        
        <button
          onClick={onDelete}
          disabled={isDeleting}
          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded-md transition-all"
          title="Delete conversation"
        >
          <Trash2 className="w-3 h-3 text-red-500" />
        </button>
      </div>
    </div>
  );
};

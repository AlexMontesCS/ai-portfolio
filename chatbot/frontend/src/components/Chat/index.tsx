import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, Plus, MessageSquare, X, EyeOff, Eye } from 'lucide-react';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { ChatMessage as ChatMessageType, Citation } from '../../types';
import { Button, LoadingSpinner } from '../ui';
import { useCurrentConversation, useCreateConversation } from '../../hooks/useApi';
import { ConversationHistory } from '../ConversationHistory';

interface ChatProps {
  conversationId: string | null;
  onConversationChange: (conversationId: string | null) => void;
  onToggleHistory?: () => void;
  historyVisible?: boolean;
}

export const Chat: React.FC<ChatProps> = ({ 
  conversationId, 
  onConversationChange, 
  onToggleHistory,
  historyVisible = true 
}) => {
  const [message, setMessage] = useState('');
  const [showMobileConversations, setShowMobileConversations] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const createConversationMutation = useCreateConversation();

  const {
    messages,
    isLoading,
    sendMessage,
    isSending,
    sendError,
  } = useCurrentConversation(conversationId);

  // Create a new conversation when component mounts if no conversationId
  useEffect(() => {
    if (!conversationId && !createConversationMutation.isLoading) {
      createConversationMutation.mutate(undefined, {
        onSuccess: (newConversation) => {
          onConversationChange(newConversation.conversation_id);
        },
        onError: (error) => {
          console.error('Failed to create conversation:', error);
        },
      });
    }
  }, [conversationId, createConversationMutation, onConversationChange]);

  const handleNewConversation = () => {
    // Clear the current conversation first
    onConversationChange(null);
    
    // Then create a new conversation
    createConversationMutation.mutate(undefined, {
      onSuccess: (newConversation) => {
        onConversationChange(newConversation.conversation_id);
      },
      onError: (error) => {
        console.error('Failed to create conversation:', error);
      },
    });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim() || isSending) return;

    try {
      const response = await sendMessage(message.trim());
      setMessage('');
      
      // If this is a new conversation, update the conversation ID
      if (!conversationId && response.conversation_id) {
        onConversationChange(response.conversation_id);
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  return (
    <div className="flex flex-col h-full relative">
      {/* Mobile Conversation History Overlay */}
      {showMobileConversations && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="fixed inset-0 bg-gray-600 bg-opacity-75"
            onClick={() => setShowMobileConversations(false)}
          />
          <div className="fixed inset-y-0 right-0 w-80 bg-white shadow-xl">
            <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Conversations</h3>
              <button
                onClick={() => setShowMobileConversations(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="h-full">
              <ConversationHistory
                currentConversationId={conversationId}
                onConversationSelect={(id) => {
                  onConversationChange(id);
                  setShowMobileConversations(false);
                }}
                onNewConversation={() => {
                  handleNewConversation();
                  setShowMobileConversations(false);
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Chat Header */}
      <div className="flex-shrink-0 border-b border-gray-200 p-4 bg-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Chat Assistant</h2>
            <p className="text-sm text-gray-500">
              Ask questions about your uploaded documents
            </p>
          </div>
          <div className="flex items-center space-x-2">
            {/* Desktop history toggle button */}
            {onToggleHistory && (
              <Button
                onClick={onToggleHistory}
                variant="outline"
                size="sm"
                className="hidden lg:flex items-center space-x-2"
                title={historyVisible ? "Hide conversation history" : "Show conversation history"}
              >
                {historyVisible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                <span>{historyVisible ? 'Hide' : 'Show'} History</span>
              </Button>
            )}
            
            {/* Mobile conversations button */}
            <Button
              onClick={() => setShowMobileConversations(true)}
              variant="outline"
              size="sm"
              className="lg:hidden flex items-center space-x-2"
            >
              <MessageSquare className="w-4 h-4" />
              <span>History</span>
            </Button>
            
            <Button
              onClick={handleNewConversation}
              disabled={createConversationMutation.isLoading}
              variant="outline"
              size="sm"
              className="flex items-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>New Chat</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && !isLoading && (
          <div className="text-center py-8">
            <Bot className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Welcome to RAG Chatbot
            </h3>
            <p className="text-gray-500 max-w-md mx-auto">
              Upload some documents and start asking questions. I'll help you find relevant information from your uploaded content.
            </p>
          </div>
        )}

        {messages.map((msg, index) => (
          <ChatMessageComponent key={index} message={msg} />
        ))}

        {isSending && (
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1 bg-white rounded-lg p-4 shadow-sm">
              <div className="flex items-center space-x-2">
                <LoadingSpinner size="sm" />
                <span className="text-gray-500">Thinking...</span>
              </div>
            </div>
          </div>
        )}

        {sendError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800 text-sm">
              Error sending message: {sendError.message || 'Please try again.'}
            </p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white p-4">
        <form onSubmit={handleSendMessage} className="flex space-x-3">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 resize-none min-h-[40px] max-h-[150px]"
              rows={1}
              disabled={isSending}
            />
          </div>
          <Button
            type="submit"
            disabled={!message.trim() || isSending}
            isLoading={isSending}
            className="flex-shrink-0"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>
    </div>
  );
};

interface ChatMessageComponentProps {
  message: ChatMessageType;
}

const ChatMessageComponent: React.FC<ChatMessageComponentProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const timestamp = new Date(message.timestamp);

  return (
    <div className={`flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser ? 'bg-gray-500' : 'bg-blue-500'
      }`}>
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : ''}`}>
        <div className={`rounded-lg p-4 shadow-sm ${
          isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-white text-gray-900'
        }`}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <p className={`text-xs text-gray-500 mt-1 ${isUser ? 'text-right' : ''}`}>
          {format(timestamp, 'MMM d, yyyy h:mm a')}
        </p>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-sm font-medium text-gray-700">Sources:</p>
            <div className="space-y-1">
              {message.citations.map((citation, index) => (
                <CitationComponent key={index} citation={citation} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

interface CitationComponentProps {
  citation: Citation;
}

const CitationComponent: React.FC<CitationComponentProps> = ({ citation }) => {
  return (
    <div className="bg-gray-50 rounded-md p-3 border border-gray-200">
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-2 mb-1">
            <FileText className="w-4 h-4 text-gray-500 flex-shrink-0" />
            <span className="text-sm font-medium text-gray-900 truncate">
              {citation.document_name}
            </span>
            {citation.page_number && (
              <span className="text-xs text-gray-500 flex-shrink-0">
                Page {citation.page_number}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600 line-clamp-2 break-words">
            {citation.text_snippet}
          </p>
        </div>
        <div className="flex-shrink-0">
          <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800 whitespace-nowrap">
            {Math.round(citation.relevance_score * 100)}% match
          </span>
        </div>
      </div>
    </div>
  );
};

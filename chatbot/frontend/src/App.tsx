import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MessageSquare, FileText, Activity, Menu, X, RefreshCw } from 'lucide-react';
import { Chat } from './components/Chat';
import { DocumentUpload } from './components/DocumentUpload';
import { ConversationHistory } from './components/ConversationHistory';
import { useHealth } from './hooks/useApi';
import './index.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: true, // Enable refetch on window focus for better UX
      staleTime: 5 * 60 * 1000, // 5 minutes default stale time
    },
  },
});

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
};

const AppContent: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [conversationHistoryVisible, setConversationHistoryVisible] = useState(true);
  const [currentConversation, setCurrentConversation] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState('chat');
  const { data: health, isFetching: healthFetching } = useHealth();

  const navigation = [
    { name: 'Chat', href: '/chat', icon: MessageSquare, id: 'chat' },
    { name: 'Documents', href: '/documents', icon: FileText, id: 'documents' },
    { name: 'Health', href: '/health', icon: Activity, id: 'health' },
  ];

  const handleNavigationClick = (pageId: string) => {
    setCurrentPage(pageId);
    setSidebarOpen(false);
  };

  const handleNewConversation = () => {
    setCurrentConversation(null);
  };

  const showConversationHistory = currentPage === 'chat' && conversationHistoryVisible;

  return (
    <div className="h-screen flex bg-gray-100">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="fixed inset-0 bg-gray-600 bg-opacity-75"
            onClick={() => setSidebarOpen(false)}
          />
        </div>
      )}

      {/* Main Navigation Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200">
          <div className="flex items-center">
            <MessageSquare className="w-8 h-8 text-blue-600" />
            <span className="ml-2 text-xl font-semibold text-gray-900">
              RAG Chatbot
            </span>
          </div>
          <button
            className="lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-6 h-6 text-gray-400" />
          </button>
        </div>

        <nav className="mt-6 px-3">
          <div className="space-y-1">
            {navigation.map((item) => (
              <button
                key={item.name}
                onClick={() => handleNavigationClick(item.id)}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                  currentPage === item.id
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
              </button>
            ))}
          </div>
        </nav>

        {/* Health Status */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Status</span>
              <div className="flex items-center">
                {healthFetching && (
                  <RefreshCw className="w-3 h-3 text-gray-400 animate-spin mr-2" />
                )}
                <div className={`w-2 h-2 rounded-full ${
                  health?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'
                }`} />
              </div>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              <p>{health?.documents_count || 0} documents</p>
              <p>{health?.chunks_count || 0} chunks</p>
            </div>
          </div>
        </div>
      </div>

      {/* Conversation History Sidebar - Only shown for chat page */}
      {showConversationHistory && (
        <div className="hidden lg:flex lg:flex-col lg:w-80 bg-white border-r border-gray-200">
          <ConversationHistory
            currentConversationId={currentConversation}
            onConversationSelect={setCurrentConversation}
            onNewConversation={handleNewConversation}
          />
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {/* Top bar */}
        <div className="flex items-center justify-between h-16 px-6 bg-white border-b border-gray-200 lg:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            className="text-gray-500 hover:text-gray-700"
          >
            <Menu className="w-6 h-6" />
          </button>
          <h1 className="text-lg font-semibold text-gray-900">RAG Chatbot</h1>
          <div /> {/* Spacer */}
        </div>

        {/* Content area */}
        <main className="flex-1 overflow-hidden">
          {currentPage === 'chat' && (
            <Chat
              conversationId={currentConversation}
              onConversationChange={setCurrentConversation}
              onToggleHistory={() => setConversationHistoryVisible(!conversationHistoryVisible)}
              historyVisible={conversationHistoryVisible}
            />
          )}
          {currentPage === 'documents' && <DocumentUpload />}
          {currentPage === 'health' && <HealthStatus />}
        </main>
      </div>
    </div>
  );
};

const HealthStatus: React.FC = () => {
  const { data: health, isLoading, error, refetch, isFetching } = useHealth();

  const handleRefresh = () => {
    refetch();
  };

  if (isLoading && !health) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-20 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-lg font-medium text-red-800 mb-2">
            Health Check Failed
          </h3>
          <p className="text-red-600">Unable to connect to the backend service.</p>
          <button
            onClick={handleRefresh}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">System Health</h2>
        <button
          onClick={handleRefresh}
          disabled={isFetching}
          className={`flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            isFetching
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
          }`}
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${isFetching ? 'animate-spin' : ''}`} />
          {isFetching ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className={`w-4 h-4 rounded-full mr-3 ${
              health?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'
            }`} />
            <h3 className="text-lg font-medium text-gray-900">Overall Status</h3>
          </div>
          <p className="mt-2 text-3xl font-bold text-gray-900 capitalize">
            {health?.status || 'Unknown'}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <FileText className="w-5 h-5 text-blue-500 mr-3" />
            <h3 className="text-lg font-medium text-gray-900">Documents</h3>
          </div>
          <p className="mt-2 text-3xl font-bold text-gray-900">
            {health?.documents_count || 0}
          </p>
          <p className="text-sm text-gray-500">
            {health?.chunks_count || 0} chunks loaded
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <Activity className="w-5 h-5 text-green-500 mr-3" />
            <h3 className="text-lg font-medium text-gray-900">Vector Store</h3>
          </div>
          <p className="mt-2 text-lg font-medium text-gray-900 capitalize">
            {health?.vector_store_status || 'Unknown'}
          </p>
          <p className="text-sm text-gray-500">Storage backend</p>
        </div>
      </div>

      <div className="mt-8 bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">System Information</h3>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <dt className="text-sm font-medium text-gray-500">Version</dt>
            <dd className="text-sm text-gray-900">{health?.version || 'Unknown'}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Last Updated</dt>
            <dd className="text-sm text-gray-900">
              {new Date().toLocaleString()}
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
};

export default App;

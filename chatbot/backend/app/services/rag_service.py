import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from ..models.schemas import (
    ChatMessage, ChatRole, Citation, ChatRequest, ChatResponse,
    ConversationResponse
)
from ..core.config import settings
from .document_service import DocumentService
import logging

logger = logging.getLogger(__name__)


class MockChatModel(BaseChatModel):
    """Mock chat model for development/testing when no API key is available."""
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Simple mock response
        response_text = "This is a mock response. Please configure a valid API key (Google Gemini or OpenAI) to get real AI responses."
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


class RAGService:
    """Service for Retrieval-Augmented Generation chat functionality."""
    
    # Class variable to store conversations across all instances
    _conversations: Dict[str, List[ChatMessage]] = {}
    
    def __init__(self, vector_store=None):
        self.document_service = DocumentService(vector_store=vector_store)
        self.llm = self._create_chat_model()
        # Use the class variable instead of instance variable
        self.conversations = RAGService._conversations
        
    def _create_chat_model(self):
        """Create chat model based on the configured provider."""
        try:
            if settings.chat_model_provider == "google" and settings.google_api_key and settings.google_api_key != "test_key_for_now":
                from openai import OpenAI
                # Use OpenAI client with Google's OpenAI-compatible API
                client = OpenAI(
                    api_key=settings.google_api_key,
                    base_url=settings.google_api_base
                )
                return ChatOpenAI(
                    model=settings.chat_model_name,
                    temperature=settings.temperature,
                    openai_api_key=settings.google_api_key,
                    openai_api_base=settings.google_api_base,
                    client=client
                )
            elif settings.chat_model_provider == "openai" and settings.openai_api_key:
                # Use OpenAI
                return ChatOpenAI(
                    model=settings.chat_model_name or settings.chat_model,
                    temperature=settings.temperature,
                    openai_api_key=settings.openai_api_key
                )
            else:
                # Fallback to a mock model for development
                logger.warning("No valid API key found, using mock chat model")
                return MockChatModel()
        except Exception as e:
            logger.warning(f"Failed to create configured chat model: {e}, using mock model")
            return MockChatModel()
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self._load_conversations()
    
    def _load_conversations(self):
        """Load conversation history from file."""
        # For simplicity, we'll store conversations in memory
        # In production, you'd want to use a proper database
        pass
    
    def _save_conversations(self):
        """Save conversation history to file."""
        # In production, save to database
        pass
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the RAG chatbot."""
        return """You are an intelligent assistant that answers questions based on the provided context from uploaded documents. 

**Instructions:**
1. Use ONLY the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite your sources by referencing the document name and page number when available
4. Provide comprehensive but concise answers
5. If multiple documents contain relevant information, synthesize the information appropriately
6. When quoting directly, use quotation marks and provide specific citations

**Response Format:**
- Start with a direct answer to the question
- Support your answer with evidence from the context
- End with citations in the format: (Document: filename, Page: X)

**Context Information:**
{context}

**Previous Conversation:**
{chat_history}

**Current Question:** {question}

Please provide a helpful and accurate response based on the available context."""
    
    def _create_citations(self, search_results: List[Dict[str, Any]]) -> List[Citation]:
        """Create citation objects from search results."""
        citations = []
        for result in search_results:
            citation = Citation(
                document_id=result.get("document_id", ""),
                document_name=result.get("filename", "Unknown"),
                page_number=result.get("page_number"),
                chunk_id=result.get("chunk_id", ""),
                relevance_score=result.get("relevance_score", 0.0),
                text_snippet=result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
            )
            citations.append(citation)
        return citations
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the LLM."""
        if not search_results:
            return "No relevant context found in the uploaded documents."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            filename = result.get("filename", "Unknown Document")
            page_info = f" (Page {result.get('page_number')})" if result.get('page_number') else ""
            content = result.get("content", "").strip()
            
            context_part = f"""
**Source {i}: {filename}{page_info}**
{content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _format_chat_history(self, messages: List[ChatMessage]) -> str:
        """Format chat history for context."""
        if not messages:
            return "No previous conversation."
        
        history_parts = []
        for msg in messages[-6:]:  # Last 6 messages for context
            role = "User" if msg.role == ChatRole.USER else "Assistant"
            history_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(history_parts)
    
    def _extract_search_query(self, message: str) -> str:
        """
        Extracts key entities from the user's message to create a more effective search query.
        It prioritizes multi-word capitalized phrases as likely proper nouns.
        """
        import re
        
        message_clean = message.strip()
        
        # Find potential multi-word entities (e.g., "Teaching Lab Studio")
        # This regex finds sequences of capitalized words.
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', message)
        
        if entities:
            # If entities are found, use them as the primary query
            final_query = ' '.join(entities)
            logger.info(f"Entities extracted for query: '{final_query}' from message: '{message}'")
        else:
            # Otherwise, use the original cleaned message
            final_query = message_clean
            logger.info(f"No entities found. Using original message for query: '{message}'")
            
        return final_query
    
    def _perform_enhanced_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Performs a multi-strategy search to improve recall and relevance.
        Combines semantic, keyword, and bigram searches, then deduplicates results.
        """
        # Strategy 1: Semantic search with the original or reformulated query
        logger.info(f"Performing semantic search with query: '{query}'")
        all_results = self.document_service.search_documents(query=query, top_k=top_k)

        # Strategy 2: Keyword-based search for better term matching
        query_words = [word for word in query.lower().split() if len(word) > 3]
        if len(query_words) >= 2:
            for word in query_words:
                logger.info(f"Performing keyword search with: '{word}'")
                all_results.extend(self.document_service.search_documents(query=word, top_k=top_k // 2))

        # Strategy 3: Bigram search for phrases if initial results are weak
        max_score = max((r.get("relevance_score", 0) for r in all_results), default=0)
        if max_score < 0.7 and len(query_words) >= 2:
            for i in range(len(query_words) - 1):
                bigram = f"{query_words[i]} {query_words[i+1]}"
                logger.info(f"Performing bigram search with: '{bigram}'")
                all_results.extend(self.document_service.search_documents(query=bigram, top_k=top_k // 2))

        # Deduplicate results, keeping the one with the highest relevance score
        seen_chunks = {}
        for result in all_results:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                current_score = result.get("relevance_score", 0.0)
                if chunk_id not in seen_chunks or current_score > seen_chunks[chunk_id].get("relevance_score", 0.0):
                    seen_chunks[chunk_id] = result
        
        # Sort unique results by score
        unique_results = sorted(seen_chunks.values(), key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        logger.info(f"Enhanced search found {len(unique_results)} unique chunks from {len(all_results)} total results.")
        return unique_results[:top_k]
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request with RAG."""
        start_time = time.time()
        
        try:
            # Get or create conversation
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation_history = self.conversations.get(conversation_id, [])
            
            # Extract key search terms from the user message for better search results
            search_query = self._extract_search_query(request.message)
            
            # Perform enhanced search with query variations
            search_results = self._perform_enhanced_search(search_query, top_k=8)
            
            # Debug logging to see what we found
            logger.info(f"Enhanced search returned {len(search_results)} results")
            for i, result in enumerate(search_results[:3]):  # Log top 3 results
                score = result.get("relevance_score", 0.0)
                content_preview = result.get("content", "")[:100]
                has_teaching_lab = "teaching lab" in result.get("content", "").lower()
                logger.info(f"Result {i+1}: score={score:.3f}, has_teaching_lab={has_teaching_lab}, preview='{content_preview}'")
            
            # Create context and chat history
            context = self._format_context(search_results)
            chat_history = self._format_chat_history(conversation_history)
            
            # Create prompt
            system_prompt = self._create_system_prompt()
            
            # Prepare messages for the LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
Context: {context}

Chat History: {chat_history}

Question: {request.message}
""")
            ]
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            
            # Create citations if requested
            citations = []
            if request.include_citations and search_results:
                citations = self._create_citations(search_results)
            
            # Create chat messages
            user_message = ChatMessage(
                role=ChatRole.USER,
                content=request.message,
                timestamp=datetime.utcnow()
            )
            
            assistant_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=response_content,
                timestamp=datetime.utcnow(),
                citations=citations,
                metadata={
                    "search_results_count": len(search_results),
                    "model_used": settings.chat_model,
                    "temperature": request.temperature or settings.temperature
                }
            )
            
            # Update conversation history
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            self.conversations[conversation_id].extend([user_message, assistant_message])
            
            # Keep only last 20 messages to prevent memory issues
            if len(self.conversations[conversation_id]) > 20:
                self.conversations[conversation_id] = self.conversations[conversation_id][-20:]
            
            processing_time = time.time() - start_time
            
            logger.info(f"Chat request processed in {processing_time:.2f}s")
            
            return ChatResponse(
                message=assistant_message,
                conversation_id=conversation_id,
                tokens_used=None,  # Would need to calculate this from the response
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            # Return error message
            error_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                timestamp=datetime.utcnow()
            )
            
            return ChatResponse(
                message=error_message,
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                processing_time=time.time() - start_time
            )
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationResponse]:
        """Get conversation history."""
        try:
            if conversation_id not in self.conversations:
                return None
            
            messages = self.conversations[conversation_id]
            if not messages:
                return None
            
            return ConversationResponse(
                conversation_id=conversation_id,
                messages=messages,
                created_at=messages[0].timestamp,
                updated_at=messages[-1].timestamp
            )
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return None
    
    def create_conversation(self) -> ConversationResponse:
        """Create a new empty conversation."""
        try:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
            
            return ConversationResponse(
                conversation_id=conversation_id,
                messages=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    def get_conversations(self) -> List[ConversationResponse]:
        """Get all conversation histories."""
        try:
            conversations = []
            for conv_id, messages in self.conversations.items():
                if messages:
                    conversations.append(ConversationResponse(
                        conversation_id=conv_id,
                        messages=messages,
                        created_at=messages[0].timestamp,
                        updated_at=messages[-1].timestamp
                    ))
            
            # Sort by last updated, newest first
            conversations.sort(key=lambda x: x.updated_at, reverse=True)
            return conversations
        except Exception as e:
            logger.error(f"Error getting conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                logger.info(f"Deleted conversation: {conversation_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    def summarize_document(self, document_id: str) -> Optional[str]:
        """Generate a summary of a specific document."""
        try:
            # Get document metadata
            document = self.document_service.get_document(document_id)
            if not document:
                return None
            
            # Search for all chunks of this document
            search_results = self.document_service.search_documents(
                query=f"document_id:{document_id}",
                top_k=10
            )
            
            if not search_results:
                return None
            
            # Combine all chunks
            content = "\n\n".join([result.get("content", "") for result in search_results])
            
            # Create summarization prompt
            summary_prompt = f"""
Please provide a comprehensive summary of the following document content:

Document Name: {document.filename}
Document Type: {document.document_type}

Content:
{content}

Provide a summary that includes:
1. Main topics and themes
2. Key points and findings
3. Important details and data
4. Overall purpose and conclusions

Summary:
"""
            
            # Get summary from LLM
            messages = [HumanMessage(content=summary_prompt)]
            response = self.llm.invoke(messages)
            
            logger.info(f"Generated summary for document: {document.filename}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error summarizing document {document_id}: {e}")
            return None

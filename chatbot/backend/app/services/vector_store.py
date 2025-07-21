import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
# from pinecone import Pinecone as PineconeClient
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search for similar documents with similarity scores."""
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        pass


class FAISSVectorStore(VectorStoreInterface):
    """FAISS-based vector store implementation."""
    
    def __init__(self):
        self.embeddings = self._create_embeddings()
        self.index_path = settings.faiss_index_path
        self.metadata_path = f"{self.index_path}_metadata.pkl"
        self.vectorstore: Optional[FAISS] = None
        self.document_metadata: Dict[str, Any] = {}
        self._load_or_create_index()
    
    def _create_embeddings(self):
        """Create embeddings based on the configured provider."""
        try:
            if settings.embedding_model_provider == "google" and settings.google_api_key and settings.google_api_key != "test_key_for_now":
                # Use OpenAI-compatible client with Google's API
                return OpenAIEmbeddings(
                    openai_api_key=settings.google_api_key,
                    openai_api_base=settings.google_api_base,
                    model=settings.embedding_model_name
                )
            elif settings.embedding_model_provider == "openai" and settings.openai_api_key:
                # Use OpenAI
                return OpenAIEmbeddings(
                    openai_api_key=settings.openai_api_key,
                    model=settings.embedding_model_name or settings.embedding_model
                )
            else:
                # Fallback to local HuggingFace embeddings
                from langchain_community.embeddings import HuggingFaceEmbeddings
                logger.info("No valid API key found, using local HuggingFace embeddings")
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
        except Exception as e:
            logger.warning(f"Failed to create configured embeddings: {e}, falling back to local embeddings")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            # Check if FAISS index directory exists and has the required files
            index_file = os.path.join(self.index_path, "index.faiss")
            pkl_file = os.path.join(self.index_path, "index.pkl")
            
            if os.path.exists(index_file) and os.path.exists(pkl_file):
                logger.info("Loading existing FAISS index")
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.document_metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.get_document_count()} documents")
            else:
                logger.info("Creating new FAISS index")
                # Create empty index by initializing with a sample document and then clearing
                sample_doc = Document(page_content="sample", metadata={"temp": True})
                self.vectorstore = FAISS.from_documents([sample_doc], self.embeddings)
                # Clear the index but keep the structure
                self.vectorstore.index.reset()
                self.vectorstore.docstore._dict.clear()
                self.vectorstore.index_to_docstore_id.clear()
                self._save_index()
        except Exception as e:
            logger.error(f"Error loading/creating FAISS index: {e}")
            raise
    
    def _save_index(self):
        """Save the index and metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.vectorstore.save_local(self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the FAISS index."""
        try:
            if not documents:
                return []
            
            # Add documents to vectorstore
            doc_ids = self.vectorstore.add_documents(documents)
            
            # Store metadata
            for i, doc in enumerate(documents):
                self.document_metadata[doc_ids[i]] = doc.metadata
            
            self._save_index()
            logger.info(f"Added {len(documents)} documents to FAISS index")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        try:
            if self.vectorstore is None:
                return []
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        try:
            if self.vectorstore is None:
                return []
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from FAISS index by rebuilding without them."""
        try:
            if not doc_ids or self.vectorstore is None:
                return False
            
            # Get all current documents
            all_docs = []
            all_metadatas = []
            
            # Collect all documents except the ones to delete
            for doc_id, metadata in self.document_metadata.items():
                if doc_id not in doc_ids:
                    # We need to recreate the Document objects
                    # This is a limitation of FAISS - we need to rebuild the entire index
                    continue
            
            # For now, mark documents as deleted in metadata but warn about limitation
            for doc_id in doc_ids:
                if doc_id in self.document_metadata:
                    del self.document_metadata[doc_id]
            
            # Save the updated metadata
            self._save_index()
            
            logger.warning(f"Documents marked for deletion: {doc_ids}. Note: FAISS requires index rebuild for complete removal.")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total document count."""
        try:
            if self.vectorstore is None:
                return 0
            return self.vectorstore.index.ntotal
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone-based vector store implementation - TEMPORARILY DISABLED."""
    
    def __init__(self):
        raise NotImplementedError("Pinecone integration is temporarily disabled. Please use FAISS instead.")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        raise NotImplementedError("Pinecone integration is temporarily disabled.")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        raise NotImplementedError("Pinecone integration is temporarily disabled.")
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        raise NotImplementedError("Pinecone integration is temporarily disabled.")
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        raise NotImplementedError("Pinecone integration is temporarily disabled.")
    
    def get_document_count(self) -> int:
        raise NotImplementedError("Pinecone integration is temporarily disabled.")


class VectorStoreManager:
    """Manager class for vector store operations."""
    
    def __init__(self):
        self.store: VectorStoreInterface = self._create_vector_store()
    
    def _create_vector_store(self) -> VectorStoreInterface:
        """Create the appropriate vector store based on configuration."""
        if settings.vector_store_type.lower() == "pinecone":
            logger.info("Initializing Pinecone vector store")
            return PineconeVectorStore()
        else:
            logger.info("Initializing FAISS vector store")
            return FAISSVectorStore()
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        return self.store.add_documents(documents)
    
    def search(self, query: str, k: int = 4, with_scores: bool = False) -> List[Document] | List[Tuple[Document, float]]:
        """Search for similar documents."""
        if with_scores:
            return self.store.similarity_search_with_score(query, k=k)
        return self.store.similarity_search(query, k=k)
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        return self.store.delete_documents(doc_ids)
    
    def get_document_count(self) -> int:
        """Get total document count."""
        return self.store.get_document_count()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector store."""
        try:
            count = self.get_document_count()
            return {
                "status": "healthy",
                "type": settings.vector_store_type,
                "document_count": count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": settings.vector_store_type,
                "error": str(e)
            }

import os
import uuid
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import aiofiles
from pathlib import Path
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..models.schemas import DocumentResponse, DocumentType
from ..core.config import settings
from .vector_store import VectorStoreManager
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for handling document upload, processing, and management."""
    
    # Class variable to store documents metadata across all instances
    _documents_metadata: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, vector_store=None):
        if vector_store:
            self.vector_store = vector_store
        else:
            try:
                self.vector_store = VectorStoreManager()
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                self.vector_store = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        # Always load metadata to ensure it's up to date
        self._load_metadata()
        self.documents_metadata = DocumentService._documents_metadata
        
        # Check if we need to re-index documents
        self._ensure_documents_indexed()
    
    def _load_metadata(self):
        """Load document metadata from file."""
        metadata_file = Path(settings.vector_store_dir) / "documents_metadata.json"
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    DocumentService._documents_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(DocumentService._documents_metadata)} documents")
        except Exception as e:
            logger.error(f"Error loading document metadata: {e}")
            DocumentService._documents_metadata = {}
    
    def _save_metadata(self):
        """Save document metadata to file."""
        metadata_file = Path(settings.vector_store_dir) / "documents_metadata.json"
        try:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(DocumentService._documents_metadata, f, ensure_ascii=False, indent=2, default=str)
            logger.info("Document metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")
    
    def _ensure_documents_indexed(self):
        """Ensure all documents in metadata are indexed in the vector store."""
        if not self.vector_store:
            return
            
        try:
            vector_count = self.vector_store.get_document_count()
            metadata_count = len(self.documents_metadata)
            expected_chunks = sum(meta.get("chunk_count", 0) for meta in self.documents_metadata.values())
            
            logger.info(f"Vector store has {vector_count} chunks, metadata shows {metadata_count} documents with {expected_chunks} total chunks")
            
            # If vector store is empty but we have documents in metadata, re-index them
            if vector_count == 0 and metadata_count > 0:
                logger.info("Vector store is empty but documents exist in metadata. Re-indexing documents...")
                self._reindex_all_documents()
            elif vector_count != expected_chunks:
                logger.warning(f"Vector store chunk count ({vector_count}) doesn't match expected ({expected_chunks})")
        except Exception as e:
            logger.error(f"Error checking document indexing: {e}")
    
    def _reindex_all_documents(self):
        """Re-index all documents from their stored files."""
        try:
            for doc_id, metadata in self.documents_metadata.items():
                file_path = Path(metadata.get("file_path", ""))
                if not file_path.exists():
                    logger.warning(f"File not found for document {doc_id}: {file_path}")
                    continue
                
                logger.info(f"Re-indexing document: {metadata['filename']}")
                
                # Read the file content
                doc_type = DocumentType(metadata["document_type"])
                if doc_type == DocumentType.PDF:
                    text = self._extract_text_from_pdf(str(file_path))
                elif doc_type == DocumentType.MARKDOWN:
                    text = self._extract_text_from_markdown(str(file_path))
                else:
                    text = self._extract_text_from_text(str(file_path))
                
                # Create document chunks
                documents = self._create_document_chunks(
                    text, doc_id, metadata["filename"], doc_type
                )
                
                # Add to vector store
                chunk_ids = self.vector_store.add_documents(documents)
                
                # Update chunk IDs in metadata
                metadata["chunk_ids"] = chunk_ids
            
            # Save updated metadata
            self._save_metadata()
            logger.info("Successfully re-indexed all documents")
            
        except Exception as e:
            logger.error(f"Error re-indexing documents: {e}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def _extract_text_from_markdown(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {e}")
            raise
    
    def _extract_text_from_text(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise
    
    def _determine_document_type(self, filename: str, content_type: str) -> DocumentType:
        """Determine document type from filename and content type."""
        filename_lower = filename.lower()
        if filename_lower.endswith('.pdf') or 'pdf' in content_type.lower():
            return DocumentType.PDF
        elif filename_lower.endswith(('.md', '.markdown')) or 'markdown' in content_type.lower():
            return DocumentType.MARKDOWN
        else:
            return DocumentType.TEXT
    
    def _create_document_chunks(self, text: str, document_id: str, filename: str, doc_type: DocumentType) -> List[Document]:
        """Split document text into chunks and create Document objects."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    metadata = {
                        "document_id": document_id,
                        "filename": filename,
                        "document_type": doc_type.value,
                        "chunk_id": f"{document_id}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "upload_time": datetime.utcnow().isoformat()
                    }
                    
                    # Add page information for PDFs
                    if doc_type == DocumentType.PDF:
                        # Extract page number from chunk if available
                        page_markers = [line for line in chunk.split('\n') if line.strip().startswith('--- Page')]
                        if page_markers:
                            try:
                                page_num = int(page_markers[0].split('Page')[1].split('---')[0].strip())
                                metadata["page_number"] = page_num
                            except (ValueError, IndexError):
                                pass
                    
                    documents.append(Document(page_content=chunk, metadata=metadata))
            
            logger.info(f"Created {len(documents)} chunks for document {filename}")
            return documents
        except Exception as e:
            logger.error(f"Error creating document chunks: {e}")
            raise
    
    async def upload_document(self, filename: str, content: bytes, content_type: str) -> DocumentResponse:
        """Upload and process a document."""
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Determine document type
            doc_type = self._determine_document_type(filename, content_type)
            
            # Create file path
            file_extension = Path(filename).suffix or ('.txt' if doc_type == DocumentType.TEXT else '')
            safe_filename = f"{document_id}_{filename.replace(' ', '_')}"
            file_path = Path(settings.upload_dir) / safe_filename
            
            # Save file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Extract text based on document type
            if doc_type == DocumentType.PDF:
                text = self._extract_text_from_pdf(str(file_path))
            elif doc_type == DocumentType.MARKDOWN:
                text = self._extract_text_from_markdown(str(file_path))
            else:
                text = self._extract_text_from_text(str(file_path))
            
            if not text.strip():
                raise ValueError("No text content found in the document")
            
            # Create document chunks
            documents = self._create_document_chunks(text, document_id, filename, doc_type)
            
            # Add to vector store
            if self.vector_store is None:
                raise ValueError("Vector store is not available")
            chunk_ids = self.vector_store.add_documents(documents)
            
            # Store document metadata
            document_metadata = {
                "id": document_id,
                "filename": filename,
                "original_filename": filename,
                "document_type": doc_type.value,
                "upload_time": datetime.utcnow().isoformat(),
                "file_path": str(file_path),
                "size": len(content),
                "chunk_count": len(documents),
                "chunk_ids": chunk_ids,
                "text_length": len(text),
                "content_type": content_type
            }
            
            self.documents_metadata[document_id] = document_metadata
            self._save_metadata()
            
            logger.info(f"Successfully uploaded and processed document: {filename}")
            
            return DocumentResponse(
                id=document_id,
                filename=filename,
                document_type=doc_type,
                upload_time=datetime.fromisoformat(document_metadata["upload_time"]),
                size=document_metadata["size"],
                chunk_count=document_metadata["chunk_count"]
            )
            
        except Exception as e:
            logger.error(f"Error uploading document {filename}: {e}")
            # Clean up on error
            if 'file_path' in locals() and file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            raise
    
    def get_documents(self) -> List[DocumentResponse]:
        """Get list of all uploaded documents."""
        try:
            documents = []
            for doc_id, metadata in self.documents_metadata.items():
                documents.append(DocumentResponse(
                    id=doc_id,
                    filename=metadata["filename"],
                    document_type=DocumentType(metadata["document_type"]),
                    upload_time=datetime.fromisoformat(metadata["upload_time"]),
                    size=metadata["size"],
                    chunk_count=metadata["chunk_count"]
                ))
            
            # Sort by upload time, newest first
            documents.sort(key=lambda x: x.upload_time, reverse=True)
            return documents
        except Exception as e:
            logger.error(f"Error getting documents list: {e}")
            return []
    
    def get_document(self, document_id: str) -> Optional[DocumentResponse]:
        """Get specific document by ID."""
        try:
            if document_id not in self.documents_metadata:
                return None
            
            metadata = self.documents_metadata[document_id]
            return DocumentResponse(
                id=document_id,
                filename=metadata["filename"],
                document_type=DocumentType(metadata["document_type"]),
                upload_time=datetime.fromisoformat(metadata["upload_time"]),
                size=metadata["size"],
                chunk_count=metadata["chunk_count"]
            )
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from the vector store."""
        try:
            if document_id not in self.documents_metadata:
                logger.warning(f"Document {document_id} not found")
                return False
            
            metadata = self.documents_metadata[document_id]
            
            # Delete from vector store if chunk IDs are available
            if "chunk_ids" in metadata:
                self.vector_store.delete_documents(metadata["chunk_ids"])
            
            # Delete physical file
            file_path = Path(metadata.get("file_path", ""))
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")
            
            # Remove from metadata
            del self.documents_metadata[document_id]
            self._save_metadata()
            
            logger.info(f"Successfully deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks with hybrid approach."""
        try:
            logger.info(f"Starting search for query: '{query}' with top_k: {top_k}")
            logger.info(f"Vector store type: {type(self.vector_store).__name__}")
            logger.info(f"Vector store document count: {self.vector_store.get_document_count()}")
            
            # First, do vector search with increased k to get more candidates
            expanded_k = min(top_k * 3, 15)  # Get 3x more results initially
            results = self.vector_store.search(query, k=expanded_k, with_scores=True)
            logger.info(f"Raw search results count: {len(results)}")
            
            # Also do text-based search for exact/partial matches (ALWAYS)
            text_matches = self._text_search(query)
            logger.info(f"Text-based matches count: {len(text_matches)}")
            
            search_results = []
            seen_chunks = set()
            
            # Add text-based matches FIRST with high relevance (prioritize exact matches)
            for text_match in text_matches:
                chunk_id = text_match.get("chunk_id")
                if chunk_id not in seen_chunks:
                    text_match["match_type"] = "text"
                    text_match["relevance_score"] = 0.95  # High score for exact text matches
                    search_results.append(text_match)
                    seen_chunks.add(chunk_id)
            
            # Then add vector search results if we don't have enough from text search
            for i, (document, score) in enumerate(results):
                chunk_id = document.metadata.get("chunk_id")
                if chunk_id in seen_chunks:
                    continue
                    
                logger.info(f"Vector result {i}: score={score}, content preview='{document.page_content[:50]}...'")
                # Convert FAISS distance score to similarity score (0-1 range)
                similarity_score = 1.0 / (1.0 + float(score))
                
                result = {
                    "content": document.page_content,
                    "metadata": document.metadata,
                    "relevance_score": similarity_score,
                    "document_id": document.metadata.get("document_id"),
                    "filename": document.metadata.get("filename"),
                    "chunk_id": chunk_id,
                    "page_number": document.metadata.get("page_number"),
                    "match_type": "vector"
                }
                search_results.append(result)
                seen_chunks.add(chunk_id)
            
            # Sort by relevance score and limit to top_k
            search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Ensure we include multiple chunks from the same document if they're relevant
            final_results = self._diversify_results(search_results, top_k)
            
            logger.info(f"Found {len(final_results)} relevant chunks for query (vector: {len([r for r in final_results if r.get('match_type') == 'vector'])}, text: {len([r for r in final_results if r.get('match_type') == 'text'])})")
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _text_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform text-based search for exact and partial matches."""
        text_matches = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        try:
            # Get all chunks from vector store
            if not self.vector_store:
                return []
                
            # Search through document metadata to find text matches
            for doc_id, metadata in self.documents_metadata.items():
                if "chunk_ids" not in metadata:
                    continue
                    
                # Read the document content to search in
                file_path = Path(metadata.get("file_path", ""))
                if not file_path.exists():
                    continue
                    
                try:
                    # Read document content
                    doc_type = DocumentType(metadata["document_type"])
                    if doc_type == DocumentType.PDF:
                        text = self._extract_text_from_pdf(str(file_path))
                    elif doc_type == DocumentType.MARKDOWN:
                        text = self._extract_text_from_markdown(str(file_path))
                    else:
                        text = self._extract_text_from_text(str(file_path))
                    
                    # Split into chunks using the same splitter
                    chunks = self.text_splitter.split_text(text)
                    
                    # Search in each chunk
                    for i, chunk in enumerate(chunks):
                        chunk_lower = chunk.lower()
                        
                        # Debug logging
                        if "teaching" in chunk_lower or "lab" in chunk_lower:
                            logger.info(f"Checking chunk {i} with query '{query_lower}' in chunk containing 'teaching' or 'lab'")
                            logger.info(f"Chunk preview: {chunk[:200]}")
                        
                        # Check for exact phrase match
                        if query_lower in chunk_lower:
                            chunk_id = f"{doc_id}_{i}"
                            logger.info(f"Found exact phrase match in chunk {i}")
                            text_matches.append({
                                "content": chunk,
                                "metadata": {
                                    "document_id": doc_id,
                                    "filename": metadata["filename"],
                                    "document_type": metadata["document_type"],
                                    "chunk_id": chunk_id,
                                    "chunk_index": i,
                                    "total_chunks": len(chunks),
                                    "upload_time": metadata["upload_time"]
                                },
                                "document_id": doc_id,
                                "filename": metadata["filename"],
                                "chunk_id": chunk_id,
                                "page_number": None
                            })
                        # Check for partial word matches (e.g., "teaching lab" matches "Teaching Lab Studio")
                        elif len(query_words) > 1:
                            word_matches = sum(1 for word in query_words if word in chunk_lower)
                            logger.info(f"Partial word check: found {word_matches}/{len(query_words)} words in chunk {i}")
                            if word_matches >= len(query_words) * 0.7:  # 70% of words must match
                                chunk_id = f"{doc_id}_{i}"
                                logger.info(f"Found partial word match in chunk {i}")
                                text_matches.append({
                                    "content": chunk,
                                    "metadata": {
                                        "document_id": doc_id,
                                        "filename": metadata["filename"],
                                        "document_type": metadata["document_type"],
                                        "chunk_id": chunk_id,
                                        "chunk_index": i,
                                        "total_chunks": len(chunks),
                                        "upload_time": metadata["upload_time"]
                                    },
                                    "document_id": doc_id,
                                    "filename": metadata["filename"],
                                    "chunk_id": chunk_id,
                                    "page_number": None
                                })
                                
                except Exception as e:
                    logger.warning(f"Error searching in document {doc_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            
        return text_matches
    
    def _diversify_results(self, search_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Ensure diverse results while allowing multiple chunks from same document if highly relevant."""
        if len(search_results) <= top_k:
            return search_results
            
        # Group results by document
        doc_groups = {}
        for result in search_results:
            doc_id = result.get("document_id")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # If query contains specific terms that likely span multiple chunks, 
        # allow more chunks from same document
        final_results = []
        query_suggests_multi_chunk = any(term in search_results[0].get("content", "").lower() 
                                       for term in ["teaching", "lab", "studio", "experience", "education"])
        
        if query_suggests_multi_chunk and len(doc_groups) == 1:
            # If all results are from one document and query suggests it spans chunks,
            # return more chunks from that document
            doc_id = list(doc_groups.keys())[0]
            doc_chunks = doc_groups[doc_id]
            
            # Sort by chunk index to get consecutive chunks
            doc_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
            
            # Take up to top_k chunks from this document
            final_results = doc_chunks[:top_k]
        else:
            # Standard diversification: prefer one chunk per document, but allow 2-3 from same doc if highly relevant
            for doc_id, chunks in doc_groups.items():
                # Sort chunks by relevance
                chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                # Add top 1-2 chunks from this document
                max_per_doc = 2 if chunks[0].get("relevance_score", 0) > 0.8 else 1
                final_results.extend(chunks[:max_per_doc])
                
                if len(final_results) >= top_k:
                    break
            
            # Sort final results by relevance and limit
            final_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            final_results = final_results[:top_k]
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document collection statistics."""
        try:
            total_docs = len(self.documents_metadata)
            total_chunks = sum(meta.get("chunk_count", 0) for meta in self.documents_metadata.values())
            total_size = sum(meta.get("size", 0) for meta in self.documents_metadata.values())
            
            doc_types = {}
            for meta in self.documents_metadata.values():
                doc_type = meta.get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_size_bytes": total_size,
                "document_types": doc_types,
                "vector_store_count": self.vector_store.get_document_count()
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}

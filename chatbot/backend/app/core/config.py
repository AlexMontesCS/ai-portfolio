from typing import Optional
from pydantic_settings import BaseSettings
import os
from pathlib import Path


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "chatbot-index"
    
    # Model Provider Configuration
    chat_model_provider: str = "google"  # openai, google
    chat_model_name: str = "gemini-1.5-flash"
    embedding_model_provider: str = "google"  # openai, google
    embedding_model_name: str = "text-embedding-004"
    
    # Vector Store Configuration
    vector_store_type: str = "faiss"  # faiss or pinecone
    faiss_index_path: str = "./vector_store/faiss_index"
    upload_dir: str = "./uploads"
    vector_store_dir: str = "./vector_store"
    
    # File Processing
    max_file_size: int = 10485760  # 10MB
    chunk_size: int = 800
    chunk_overlap: int = 400
    max_tokens: int = 4000
    
    # CORS
    cors_origins: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    debug: bool = False
    
    # Model Configuration (legacy - kept for backward compatibility)
    embedding_model: str = "text-embedding-004"
    chat_model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

"""
Application configuration settings.
"""
import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Basic app settings
    PROJECT_NAME: str = "Math Engine API"
    PROJECT_DESCRIPTION: str = "AI-Powered Math Equation Solver and Visualizer API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./math_engine.db")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379")
    
    # AI/LLM Settings
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_PROVIDER: str = "gemini"  # "openai" or "gemini"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    ENABLE_LLM_EXPLANATIONS: bool = True
    
    # Math Engine Settings
    MAX_EXPRESSION_LENGTH: int = 1000
    DEFAULT_PRECISION: int = 10
    ENABLE_STEP_BY_STEP: bool = True
    CACHE_SOLUTIONS: bool = True
    CACHE_TTL_SECONDS: int = 3600
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # This allows extra fields from .env to be ignored


# Global settings instance
settings = Settings()

"""
FastAPI application factory and configuration.
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exception_handlers import setup_exception_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    
    logger.info("Creating FastAPI application...")
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        logger.info(f"Incoming request: {request.method} {request.url}")
        logger.info(f"Headers: {dict(request.headers)}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.4f}s")
        return response

    # Security middleware
    logger.info(f"Adding TrustedHost middleware with hosts: {settings.ALLOWED_HOSTS}")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

    # CORS middleware
    logger.info(f"Adding CORS middleware with origins: {settings.BACKEND_CORS_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    logger.info("Setting up exception handlers...")
    setup_exception_handlers(app)

    # API router
    logger.info(f"Including API router with prefix: {settings.API_V1_STR}")
    app.include_router(api_router, prefix=settings.API_V1_STR)

    logger.info("FastAPI application created successfully!")
    return app


# Create app instance
app = create_application()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Powered Math Equation Solver API",
        "version": settings.VERSION,
        "docs": "/docs",
        "api": settings.API_V1_STR,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

"""
Exception handlers for the FastAPI application.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import MathEngineException


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for the application."""
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Exception",
                "message": exc.detail,
                "status_code": exc.status_code,
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": "Invalid request data",
                "details": exc.errors(),
            }
        )
    
    @app.exception_handler(MathEngineException)
    async def math_engine_exception_handler(request: Request, exc: MathEngineException):
        """Handle math engine specific exceptions."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Math Engine Error",
                "message": str(exc),
                "error_type": exc.__class__.__name__,
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
            }
        )

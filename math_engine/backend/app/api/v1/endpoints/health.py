"""
Health check and system status endpoints.
"""
import time
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.llm import LLMExplanationService
from app.models.schemas import HealthResponse

router = APIRouter()

# Store start time for uptime calculation
START_TIME = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns the current status of the API service including:
    - Service status
    - Current timestamp
    - API version
    - Service uptime
    
    **Example response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",
        "version": "1.0.0",
        "uptime_seconds": 3600.5
    }
    ```
    """
    current_time = time.time()
    uptime = current_time - START_TIME
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=settings.VERSION,
        uptime_seconds=round(uptime, 2)
    )


@router.get("/detailed")
async def detailed_health_check() -> JSONResponse:
    """
    Detailed health check with service component status.
    
    This endpoint provides comprehensive health information including:
    - API service status
    - LLM service status
    - Configuration status
    - System metrics
    
    **Example response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",
        "version": "1.0.0",
        "uptime_seconds": 3600.5,
        "services": {
            "llm": {
                "enabled": true,
                "model": "gpt-3.5-turbo",
                "api_key_configured": true
            },
            "math_engine": {
                "status": "operational"
            }
        },
        "configuration": {
            "max_expression_length": 1000,
            "default_precision": 10,
            "cache_enabled": true
        }
    }
    ```
    """
    current_time = time.time()
    uptime = current_time - START_TIME
    
    # Check LLM service status
    llm_service = LLMExplanationService()
    llm_status = llm_service.get_status()
    
    # Determine overall status
    overall_status = "healthy"
    if not llm_status["enabled"] and settings.ENABLE_LLM_EXPLANATIONS:
        overall_status = "degraded"  # LLM expected but not working
    
    health_data = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.VERSION,
        "uptime_seconds": round(uptime, 2),
        "services": {
            "llm": llm_status,
            "math_engine": {
                "status": "operational",
                "parser": "sympy",
                "solver": "sympy",
                "visualizer": "d3js"
            }
        },
        "configuration": {
            "max_expression_length": settings.MAX_EXPRESSION_LENGTH,
            "default_precision": settings.DEFAULT_PRECISION,
            "step_by_step_enabled": settings.ENABLE_STEP_BY_STEP,
            "cache_enabled": settings.CACHE_SOLUTIONS,
            "cache_ttl_seconds": settings.CACHE_TTL_SECONDS,
        },
        "api": {
            "base_url": settings.API_V1_STR,
            "docs_url": "/docs",
            "cors_origins": settings.BACKEND_CORS_ORIGINS,
        }
    }
    
    return JSONResponse(content=health_data)


@router.get("/ready")
async def readiness_check() -> JSONResponse:
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if the service is ready to accept requests,
    503 if the service is not ready.
    """
    try:
        # Test core functionality
        from app.services.parser import MathParser
        from app.services.solver import MathSolver
        
        parser = MathParser()
        solver = MathSolver()
        
        # Simple test parse and solve
        test_expr = "2 + 2"
        parsed_expr, _ = parser.parse_expression(test_expr)
        result = solver.solve_equation(test_expr, include_steps=False)
        
        return JSONResponse(
            content={
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "test_result": "passed"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e)
            }
        )


@router.get("/live")
async def liveness_check() -> JSONResponse:
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if the service is alive, 503 if it should be restarted.
    """
    return JSONResponse(
        content={
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": round(time.time() - START_TIME, 2)
        }
    )

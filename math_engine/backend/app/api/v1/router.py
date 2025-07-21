"""
API router for v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import parse, solve, visualize, health

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(parse.router, prefix="/parse", tags=["parsing"])
api_router.include_router(solve.router, prefix="/solve", tags=["solving"])
api_router.include_router(visualize.router, prefix="/visualize", tags=["visualization"])
api_router.include_router(health.router, prefix="/health", tags=["health"])

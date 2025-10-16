"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SearchGPT",
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Add checks for dependencies (DB, cache, etc.)
    return {
        "status": "ready",
        "service": "SearchGPT",
    }

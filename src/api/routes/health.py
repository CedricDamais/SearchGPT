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
    # I would have to add more checks here before sending the response
    return {
        "status": "ready",
        "service": "SearchGPT",
    }

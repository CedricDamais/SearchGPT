"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, search

app = FastAPI(
    title="SearchGPT",
    description="LLM-powered search engine with hybrid search and re-ranking",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SearchGPT API",
        "docs": "/docs",
        "version": "0.1.0",
    }

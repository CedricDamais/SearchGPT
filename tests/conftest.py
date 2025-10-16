"""Pytest configuration and shared fixtures."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """Create a test client for the FastAPI application."""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def sample_query():
    """Sample search query for testing."""
    return "How does hybrid search work?"


@pytest.fixture
def sample_documents():
    """Sample documents for testing search functionality."""
    return [
        {
            "id": "doc1",
            "title": "Introduction to Hybrid Search",
            "content": "Hybrid search combines BM25 and vector search for better results.",
        },
        {
            "id": "doc2",
            "title": "Vector Embeddings Explained",
            "content": "Vector embeddings represent text as dense vectors in semantic space.",
        },
        {
            "id": "doc3",
            "title": "BM25 Algorithm Overview",
            "content": "BM25 is a probabilistic ranking function used in information retrieval.",
        },
    ]

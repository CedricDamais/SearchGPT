"""Tests for search API endpoints."""

import pytest


def test_search_endpoint(api_client):
    """Test the search endpoint."""
    response = api_client.post(
        "/api/v1/search",
        json={
            "query": "hybrid search",
            "top_k": 10,
            "use_reranking": True,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "total" in data
    assert "processing_time_ms" in data


def test_search_with_invalid_query(api_client):
    """Test search with empty query."""
    response = api_client.post(
        "/api/v1/search",
        json={
            "query": "",
            "top_k": 10,
        },
    )
    
    assert response.status_code == 422  # Validation error

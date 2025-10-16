"""Tests for health check endpoints."""


def test_health_check(api_client):
    """Test the health check endpoint."""
    response = api_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SearchGPT"


def test_readiness_check(api_client):
    """Test the readiness check endpoint."""
    response = api_client.get("/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "SearchGPT"


def test_root_endpoint(api_client):
    """Test the root endpoint."""
    response = api_client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

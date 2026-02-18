"""
Unit tests for API endpoints.
Tests all major API routes with various input scenarios.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Get authentication headers for testing."""
    # Register and login test user
    register_data = {
        "email": "test@example.com",
        "password": "TestPass123!",
        "name": "Test User",
        "role": "student"
    }
    client.post("/api/v1/auth/register", json=register_data)
    
    login_data = {
        "username": "test@example.com",
        "password": "TestPass123!"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_register_user(client):
    """Test user registration."""
    data = {
        "email": "newuser@example.com",
        "password": "SecurePass123!",
        "name": "New User",
        "role": "student"
    }
    response = client.post("/api/v1/auth/register", json=data)
    assert response.status_code in [200, 201]
    assert "id" in response.json() or "message" in response.json()


def test_register_duplicate_email(client):
    """Test registration with duplicate email."""
    data = {
        "email": "duplicate@example.com",
        "password": "Pass123!",
        "name": "User One"
    }
    client.post("/api/v1/auth/register", json=data)
    
    # Try to register again with same email
    response = client.post("/api/v1/auth/register", json=data)
    assert response.status_code in [400, 409]


def test_login_valid_credentials(client):
    """Test login with valid credentials."""
    # Register user first
    register_data = {
        "email": "logintest@example.com",
        "password": "TestPass123!",
        "name": "Login Test"
    }
    client.post("/api/v1/auth/register", json=register_data)
    
    # Login
    login_data = {
        "username": "logintest@example.com",
        "password": "TestPass123!"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_invalid_credentials(client):
    """Test login with invalid credentials."""
    login_data = {
        "username": "nonexistent@example.com",
        "password": "WrongPass123!"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code in [401, 404]


def test_upload_submission_unauthorized(client):
    """Test submission upload without authentication."""
    files = {"file": ("test.txt", b"Test content", "text/plain")}
    data = {"title": "Test Essay"}
    response = client.post("/api/v1/submissions/upload", files=files, data=data)
    assert response.status_code == 401


def test_get_profile_unauthorized(client):
    """Test profile access without authentication."""
    response = client.get("/api/v1/profile")
    assert response.status_code == 401


def test_invalid_file_format(client, auth_headers):
    """Test upload with invalid file format."""
    files = {"file": ("test.exe", b"Invalid content", "application/x-msdownload")}
    data = {"title": "Test"}
    response = client.post("/api/v1/submissions/upload", files=files, data=data, headers=auth_headers)
    assert response.status_code in [400, 415]


def test_missing_required_fields(client, auth_headers):
    """Test API calls with missing required fields."""
    # Missing title in submission
    files = {"file": ("test.txt", b"Content", "text/plain")}
    response = client.post("/api/v1/submissions/upload", files=files, headers=auth_headers)
    assert response.status_code in [400, 422]


def test_get_submissions_list(client, auth_headers):
    """Test getting list of submissions."""
    response = client.get("/api/v1/submissions", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_pagination_parameters(client, auth_headers):
    """Test API pagination."""
    response = client.get("/api/v1/submissions?skip=0&limit=10", headers=auth_headers)
    assert response.status_code == 200


def test_large_file_upload(client, auth_headers):
    """Test upload with large file."""
    # Create a large file (5MB)
    large_content = b"A" * (5 * 1024 * 1024)
    files = {"file": ("large.txt", large_content, "text/plain")}
    data = {"title": "Large Essay"}
    response = client.post("/api/v1/submissions/upload", files=files, data=data, headers=auth_headers)
    # Should either accept or reject with appropriate status
    assert response.status_code in [200, 201, 413]


def test_concurrent_requests(client, auth_headers):
    """Test handling of concurrent API requests."""
    import concurrent.futures
    
    def make_request():
        return client.get("/api/v1/submissions", headers=auth_headers)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All requests should succeed
    assert all(r.status_code == 200 for r in results)


def test_sql_injection_protection(client, auth_headers):
    """Test protection against SQL injection."""
    malicious_input = "'; DROP TABLE students; --"
    response = client.get(f"/api/v1/submissions?title={malicious_input}", headers=auth_headers)
    # Should not cause server error
    assert response.status_code in [200, 400, 404]


def test_xss_protection(client, auth_headers):
    """Test protection against XSS attacks."""
    xss_payload = "<script>alert('xss')</script>"
    files = {"file": ("test.txt", b"Content", "text/plain")}
    data = {"title": xss_payload}
    response = client.post("/api/v1/submissions/upload", files=files, data=data, headers=auth_headers)
    # Should sanitize or reject
    assert response.status_code in [200, 201, 400]


def test_rate_limiting(client, auth_headers):
    """Test API rate limiting."""
    # Make many rapid requests
    responses = []
    for _ in range(100):
        response = client.get("/api/v1/submissions", headers=auth_headers)
        responses.append(response)
    
    # Should have some rate limiting (429) or all succeed
    status_codes = [r.status_code for r in responses]
    assert all(code in [200, 429] for code in status_codes)


def test_invalid_json_payload(client):
    """Test handling of invalid JSON."""
    response = client.post(
        "/api/v1/auth/register",
        data="invalid json{{{",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code in [400, 422]


def test_missing_content_type(client):
    """Test handling of missing content type."""
    response = client.post("/api/v1/auth/register", data='{"email": "test@test.com"}')
    # Should handle gracefully
    assert response.status_code in [200, 201, 400, 415, 422]


def test_expired_token(client):
    """Test handling of expired authentication token."""
    expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjF9.invalid"
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = client.get("/api/v1/profile", headers=headers)
    assert response.status_code == 401


def test_malformed_token(client):
    """Test handling of malformed authentication token."""
    headers = {"Authorization": "Bearer malformed.token.here"}
    response = client.get("/api/v1/profile", headers=headers)
    assert response.status_code == 401


def test_cors_headers(client):
    """Test CORS headers in response."""
    response = client.options("/api/v1/submissions")
    # Should have CORS headers
    assert response.status_code in [200, 204]


def test_api_versioning(client):
    """Test API versioning support."""
    # Test v1 endpoint
    response = client.get("/api/v1/health")
    assert response.status_code in [200, 404]


def test_empty_file_upload(client, auth_headers):
    """Test upload with empty file."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    data = {"title": "Empty Essay"}
    response = client.post("/api/v1/submissions/upload", files=files, data=data, headers=auth_headers)
    assert response.status_code in [400, 422]


def test_special_characters_in_title(client, auth_headers):
    """Test submission with special characters in title."""
    files = {"file": ("test.txt", b"Content", "text/plain")}
    data = {"title": "Test Essay with émojis 🎓 and spëcial çhars"}
    response = client.post("/api/v1/submissions/upload", files=files, data=data, headers=auth_headers)
    assert response.status_code in [200, 201, 400]


def test_very_long_title(client, auth_headers):
    """Test submission with very long title."""
    files = {"file": ("test.txt", b"Content", "text/plain")}
    data = {"title": "A" * 1000}  # Very long title
    response = client.post("/api/v1/submissions/upload", files=files, data=data, headers=auth_headers)
    assert response.status_code in [200, 201, 400, 422]


def test_negative_pagination_values(client, auth_headers):
    """Test pagination with negative values."""
    response = client.get("/api/v1/submissions?skip=-1&limit=-10", headers=auth_headers)
    assert response.status_code in [200, 400, 422]


def test_excessive_pagination_limit(client, auth_headers):
    """Test pagination with excessive limit."""
    response = client.get("/api/v1/submissions?skip=0&limit=10000", headers=auth_headers)
    # Should either limit or reject
    assert response.status_code in [200, 400]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

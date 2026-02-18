"""
Pytest configuration and shared fixtures for integration and E2E tests.
"""
import pytest
import asyncio
import os
from typing import AsyncGenerator

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as load test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ["TESTING"] = "1"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_integration.db"
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
    os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-testing-only"
    
    yield
    
    # Cleanup
    if os.path.exists("test_integration.db"):
        try:
            os.remove("test_integration.db")
        except:
            pass


@pytest.fixture(scope="function")
def mock_blockchain_service(monkeypatch):
    """Mock blockchain service for testing."""
    class MockBlockchainService:
        async def mint_proof_token(self, *args, **kwargs):
            return {
                "transaction_hash": "0x" + "a" * 64,
                "token_id": 1,
                "status": "success"
            }
        
        async def get_submission_proof(self, *args, **kwargs):
            return {
                "token_id": 1,
                "verified": True,
                "timestamp": "2024-01-01T00:00:00Z"
            }
    
    return MockBlockchainService()


@pytest.fixture(scope="function")
def mock_ipfs_service(monkeypatch):
    """Mock IPFS service for testing."""
    class MockIPFSService:
        async def upload_content(self, content: str):
            return "Qm" + "a" * 44  # Mock IPFS hash
        
        async def get_content(self, ipfs_hash: str):
            return "Mock content from IPFS"
    
    return MockIPFSService()


@pytest.fixture(scope="function")
def mock_ml_services(monkeypatch):
    """Mock ML/DL services for faster testing."""
    class MockVerificationService:
        async def verify_authorship(self, *args, **kwargs):
            return {
                "authorship_score": 0.85,
                "confidence": 0.90,
                "status": "PASS"
            }
        
        async def detect_ai_content(self, *args, **kwargs):
            return {
                "ai_probability": 0.15,
                "human_probability": 0.85,
                "confidence": 0.88
            }
        
        async def check_duplicates(self, *args, **kwargs):
            return {
                "is_duplicate": False,
                "matches": [],
                "highest_similarity": 0.0
            }
    
    return MockVerificationService()


# Performance test configuration
PERFORMANCE_THRESHOLDS = {
    "max_response_time": 5.0,  # seconds
    "min_success_rate": 0.90,  # 90%
    "max_p95_response_time": 10.0,  # seconds
    "min_throughput": 10,  # requests per second
}


@pytest.fixture
def performance_thresholds():
    """Provide performance thresholds for tests."""
    return PERFORMANCE_THRESHOLDS

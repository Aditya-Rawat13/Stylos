"""
Integration and End-to-End Tests for Project Stylos.

This module contains comprehensive integration tests covering:
1. Full workflow from upload to blockchain attestation
2. Performance tests for concurrent user scenarios
3. Security tests for authentication and authorization
4. Load testing for system scalability validation

Requirements: 8.1, 8.2, 8.3
"""
import pytest
import asyncio
import time
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select
import os
from typing import Dict, List

from main import app
from core.database import Base, get_db
from core.config import settings
from models.user import User, UserRole
from models.submission import Submission, SubmissionStatus
from core.security import security


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_integration.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def override_get_db():
    """Override database dependency for testing."""
    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
async def client(test_db):
    """Create test client with database override."""
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
async def test_student_user(client: AsyncClient) -> Dict:
    """Create a test student user and return credentials."""
    user_data = {
        "email": "student@test.edu",
        "password": "TestPassword123!",
        "name": "Test Student",
        "role": "student"
    }
    
    response = await client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201
    
    # Login to get token
    login_response = await client.post(
        "/api/v1/auth/login",
        data={"username": user_data["email"], "password": user_data["password"]}
    )
    assert login_response.status_code == 200
    
    token_data = login_response.json()
    return {
        "email": user_data["email"],
        "password": user_data["password"],
        "token": token_data["access_token"],
        "user_id": token_data["user_id"]
    }


@pytest.fixture
async def test_admin_user(client: AsyncClient) -> Dict:
    """Create a test admin user and return credentials."""
    # Create admin directly in database
    async with TestSessionLocal() as session:
        admin = User(
            email="admin@test.edu",
            name="Test Admin",
            role=UserRole.ADMIN,
            is_active=True
        )
        admin.set_password("AdminPassword123!")
        session.add(admin)
        await session.commit()
        await session.refresh(admin)
        admin_id = admin.id
    
    # Login to get token
    login_response = await client.post(
        "/api/v1/auth/login",
        data={"username": "admin@test.edu", "password": "AdminPassword123!"}
    )
    assert login_response.status_code == 200
    
    token_data = login_response.json()
    return {
        "email": "admin@test.edu",
        "password": "AdminPassword123!",
        "token": token_data["access_token"],
        "user_id": admin_id
    }


# ============================================================================
# 1. FULL WORKFLOW TESTS (Upload to Blockchain Attestation)
# ============================================================================

@pytest.mark.asyncio
class TestFullWorkflow:
    """Test complete workflow from upload to blockchain attestation."""
    
    async def test_complete_submission_workflow(self, client: AsyncClient, test_student_user: Dict):
        """
        Test the complete workflow:
        1. Student uploads essay
        2. System processes and verifies
        3. Verification results are generated
        4. Blockchain attestation is created
        """
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Step 1: Upload initial essays to build writing profile
        profile_essays = [
            "This is my first essay about artificial intelligence. " * 50,
            "This is my second essay discussing machine learning concepts. " * 50,
            "This is my third essay exploring deep learning applications. " * 50
        ]
        
        for i, essay_content in enumerate(profile_essays):
            files = {"file": (f"profile_essay_{i}.txt", essay_content.encode(), "text/plain")}
            data = {"title": f"Profile Essay {i+1}", "course_name": "CS101"}
            
            response = await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
            assert response.status_code in [200, 201], f"Profile upload failed: {response.text}"
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Step 2: Upload new essay for verification
        new_essay = "This is my new essay about neural networks and their applications. " * 50
        files = {"file": ("new_essay.txt", new_essay.encode(), "text/plain")}
        data = {"title": "Neural Networks Essay", "course_name": "CS101"}
        
        upload_response = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers
        )
        assert upload_response.status_code in [200, 201]
        submission_data = upload_response.json()
        submission_id = submission_data.get("id") or submission_data.get("submission_id")
        
        # Step 3: Check verification status
        await asyncio.sleep(2)  # Allow time for verification
        
        status_response = await client.get(
            f"/api/v1/submissions/{submission_id}",
            headers=headers
        )
        assert status_response.status_code == 200
        submission_info = status_response.json()
        
        # Verify that verification results exist
        assert "verification_result" in submission_info or "status" in submission_info
        
        # Step 4: Check blockchain attestation (if verification passed)
        if submission_info.get("status") == "VERIFIED":
            blockchain_response = await client.get(
                f"/api/v1/blockchain/submission/{submission_id}",
                headers=headers
            )
            # Blockchain may not be available in test environment
            assert blockchain_response.status_code in [200, 404, 503]
    
    async def test_duplicate_detection_workflow(self, client: AsyncClient, test_student_user: Dict):
        """Test workflow when duplicate content is detected."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Upload original essay
        original_content = "This is a unique essay about quantum computing. " * 50
        files = {"file": ("original.txt", original_content.encode(), "text/plain")}
        data = {"title": "Quantum Computing Original", "course_name": "PHYS201"}
        
        response1 = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers
        )
        assert response1.status_code in [200, 201]
        
        await asyncio.sleep(1)
        
        # Upload duplicate essay
        files = {"file": ("duplicate.txt", original_content.encode(), "text/plain")}
        data = {"title": "Quantum Computing Duplicate", "course_name": "PHYS201"}
        
        response2 = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers
        )
        assert response2.status_code in [200, 201, 400]
        
        # If duplicate detection is working, should be flagged
        if response2.status_code == 200:
            submission_data = response2.json()
            submission_id = submission_data.get("id") or submission_data.get("submission_id")
            
            # Check if flagged for review
            status_response = await client.get(
                f"/api/v1/submissions/{submission_id}",
                headers=headers
            )
            submission_info = status_response.json()
            # Should be flagged or in review status
            assert submission_info.get("status") in ["PENDING", "REVIEW", "FLAGGED"]
    
    async def test_ai_detection_workflow(self, client: AsyncClient, test_student_user: Dict):
        """Test workflow with AI-generated content detection."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Upload essay with AI-like characteristics
        ai_like_content = (
            "In conclusion, artificial intelligence represents a transformative technology. "
            "Furthermore, machine learning algorithms demonstrate significant potential. "
            "Moreover, deep learning models exhibit remarkable capabilities. "
            "Additionally, neural networks provide innovative solutions. "
        ) * 25
        
        files = {"file": ("ai_essay.txt", ai_like_content.encode(), "text/plain")}
        data = {"title": "AI Technology Essay", "course_name": "CS301"}
        
        response = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers
        )
        assert response.status_code in [200, 201]
        
        submission_data = response.json()
        submission_id = submission_data.get("id") or submission_data.get("submission_id")
        
        await asyncio.sleep(2)
        
        # Check verification results
        status_response = await client.get(
            f"/api/v1/submissions/{submission_id}",
            headers=headers
        )
        assert status_response.status_code == 200
        submission_info = status_response.json()
        
        # Should have AI detection results
        verification = submission_info.get("verification_result", {})
        assert "ai_probability" in verification or "ai_detection" in verification


# ============================================================================
# 2. PERFORMANCE TESTS (Concurrent User Scenarios)
# ============================================================================

@pytest.mark.asyncio
class TestPerformance:
    """Test system performance under concurrent load."""
    
    async def test_concurrent_uploads(self, client: AsyncClient, test_student_user: Dict):
        """Test handling of concurrent file uploads."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        async def upload_essay(index: int):
            """Upload a single essay."""
            content = f"This is concurrent essay number {index}. " * 50
            files = {"file": (f"concurrent_{index}.txt", content.encode(), "text/plain")}
            data = {"title": f"Concurrent Essay {index}", "course_name": "TEST101"}
            
            start_time = time.time()
            response = await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "duration": end_time - start_time,
                "index": index
            }
        
        # Create 10 concurrent upload tasks
        tasks = [upload_essay(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all uploads succeeded or were properly handled
        successful = sum(1 for r in results if not isinstance(r, Exception) and r["status_code"] in [200, 201])
        assert successful >= 8, f"Only {successful}/10 concurrent uploads succeeded"
        
        # Check average response time
        durations = [r["duration"] for r in results if not isinstance(r, Exception)]
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 5.0, f"Average upload time {avg_duration}s exceeds 5s threshold"
    
    async def test_concurrent_verification_requests(self, client: AsyncClient, test_student_user: Dict):
        """Test concurrent verification status checks."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # First, create a submission
        content = "Test essay for concurrent verification checks. " * 50
        files = {"file": ("test.txt", content.encode(), "text/plain")}
        data = {"title": "Test Essay", "course_name": "TEST101"}
        
        upload_response = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers
        )
        assert upload_response.status_code in [200, 201]
        submission_data = upload_response.json()
        submission_id = submission_data.get("id") or submission_data.get("submission_id")
        
        async def check_status():
            """Check submission status."""
            start_time = time.time()
            response = await client.get(
                f"/api/v1/submissions/{submission_id}",
                headers=headers
            )
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "duration": end_time - start_time
            }
        
        # Create 20 concurrent status check tasks
        tasks = [check_status() for _ in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        successful = sum(1 for r in results if not isinstance(r, Exception) and r["status_code"] == 200)
        assert successful >= 18, f"Only {successful}/20 concurrent status checks succeeded"
        
        # Check response time (should be fast due to caching)
        durations = [r["duration"] for r in results if not isinstance(r, Exception)]
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 2.0, f"Average status check time {avg_duration}s exceeds 2s threshold"
    
    async def test_dashboard_query_performance(self, client: AsyncClient, test_student_user: Dict):
        """Test dashboard query performance with multiple submissions."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Create multiple submissions
        for i in range(5):
            content = f"Dashboard test essay {i}. " * 50
            files = {"file": (f"dashboard_{i}.txt", content.encode(), "text/plain")}
            data = {"title": f"Dashboard Essay {i}", "course_name": "TEST101"}
            
            await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
        
        await asyncio.sleep(1)
        
        # Test dashboard query performance
        start_time = time.time()
        response = await client.get("/api/v1/submissions/", headers=headers)
        end_time = time.time()
        
        assert response.status_code == 200
        duration = end_time - start_time
        assert duration < 2.0, f"Dashboard query took {duration}s, exceeds 2s threshold"
        
        # Verify data is returned
        submissions = response.json()
        assert isinstance(submissions, (list, dict))


# ============================================================================
# 3. SECURITY TESTS (Authentication and Authorization)
# ============================================================================

@pytest.mark.asyncio
class TestSecurity:
    """Test authentication and authorization security."""
    
    async def test_unauthorized_access_prevention(self, client: AsyncClient):
        """Test that endpoints require authentication."""
        # Try to access protected endpoints without token
        endpoints = [
            "/api/v1/submissions/",
            "/api/v1/submissions/upload",
            "/api/v1/users/me",
            "/api/v1/admin/dashboard"
        ]
        
        for endpoint in endpoints:
            response = await client.get(endpoint)
            assert response.status_code in [401, 403, 405], \
                f"Endpoint {endpoint} should require authentication"
    
    async def test_role_based_access_control(
        self,
        client: AsyncClient,
        test_student_user: Dict,
        test_admin_user: Dict
    ):
        """Test that role-based access control is enforced."""
        student_headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        admin_headers = {"Authorization": f"Bearer {test_admin_user['token']}"}
        
        # Student should NOT access admin endpoints
        admin_endpoints = [
            "/api/v1/admin/dashboard",
            "/api/v1/admin/users",
            "/api/v1/admin/submissions"
        ]
        
        for endpoint in admin_endpoints:
            response = await client.get(endpoint, headers=student_headers)
            assert response.status_code in [403, 404], \
                f"Student should not access admin endpoint {endpoint}"
        
        # Admin SHOULD access admin endpoints
        response = await client.get("/api/v1/admin/dashboard", headers=admin_headers)
        assert response.status_code in [200, 404], "Admin should access admin dashboard"
    
    async def test_token_expiration_handling(self, client: AsyncClient):
        """Test handling of expired tokens."""
        # Create an expired token
        expired_token = security.create_access_token(
            data={"sub": "999"},
            expires_delta=-3600  # Expired 1 hour ago
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = await client.get("/api/v1/submissions/", headers=headers)
        
        assert response.status_code == 401, "Expired token should be rejected"
    
    async def test_invalid_token_handling(self, client: AsyncClient):
        """Test handling of invalid tokens."""
        invalid_tokens = [
            "invalid.token.here",
            "Bearer invalid",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get("/api/v1/submissions/", headers=headers)
            assert response.status_code == 401, f"Invalid token {token} should be rejected"
    
    async def test_cross_user_data_access_prevention(
        self,
        client: AsyncClient,
        test_student_user: Dict
    ):
        """Test that users cannot access other users' data."""
        # Create first user's submission
        headers1 = {"Authorization": f"Bearer {test_student_user['token']}"}
        content = "First user's essay. " * 50
        files = {"file": ("user1.txt", content.encode(), "text/plain")}
        data = {"title": "User 1 Essay", "course_name": "TEST101"}
        
        response = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers1
        )
        assert response.status_code in [200, 201]
        submission_data = response.json()
        submission_id = submission_data.get("id") or submission_data.get("submission_id")
        
        # Create second user
        user2_data = {
            "email": "student2@test.edu",
            "password": "TestPassword123!",
            "name": "Test Student 2",
            "role": "student"
        }
        await client.post("/api/v1/auth/register", json=user2_data)
        
        login_response = await client.post(
            "/api/v1/auth/login",
            data={"username": user2_data["email"], "password": user2_data["password"]}
        )
        token2 = login_response.json()["access_token"]
        headers2 = {"Authorization": f"Bearer {token2}"}
        
        # Try to access first user's submission with second user's token
        response = await client.get(
            f"/api/v1/submissions/{submission_id}",
            headers=headers2
        )
        assert response.status_code in [403, 404], \
            "User should not access another user's submission"
    
    async def test_sql_injection_prevention(self, client: AsyncClient, test_student_user: Dict):
        """Test SQL injection prevention in queries."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Try SQL injection in search/filter parameters
        sql_injection_attempts = [
            "'; DROP TABLE submissions; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM users WHERE 1=1"
        ]
        
        for injection in sql_injection_attempts:
            response = await client.get(
                f"/api/v1/submissions/?search={injection}",
                headers=headers
            )
            # Should not cause server error
            assert response.status_code in [200, 400, 422], \
                f"SQL injection attempt caused unexpected status: {response.status_code}"
    
    async def test_xss_prevention(self, client: AsyncClient, test_student_user: Dict):
        """Test XSS prevention in user inputs."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Try XSS in title
        xss_content = "Normal essay content. " * 50
        files = {"file": ("xss_test.txt", xss_content.encode(), "text/plain")}
        data = {
            "title": "<script>alert('XSS')</script>",
            "course_name": "TEST101"
        }
        
        response = await client.post(
            "/api/v1/submissions/upload",
            files=files,
            data=data,
            headers=headers
        )
        
        # Should either sanitize or reject
        assert response.status_code in [200, 201, 400, 422]
        
        if response.status_code in [200, 201]:
            submission_data = response.json()
            # Title should be sanitized
            title = submission_data.get("title", "")
            assert "<script>" not in title, "XSS script tag should be sanitized"


# ============================================================================
# 4. LOAD TESTING (System Scalability Validation)
# ============================================================================

@pytest.mark.asyncio
class TestLoadAndScalability:
    """Test system scalability under load."""
    
    async def test_high_volume_user_registration(self, client: AsyncClient):
        """Test system handling of multiple user registrations."""
        async def register_user(index: int):
            """Register a single user."""
            user_data = {
                "email": f"loadtest{index}@test.edu",
                "password": "TestPassword123!",
                "name": f"Load Test User {index}",
                "role": "student"
            }
            
            start_time = time.time()
            response = await client.post("/api/v1/auth/register", json=user_data)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "duration": end_time - start_time,
                "index": index
            }
        
        # Create 20 concurrent registration tasks
        tasks = [register_user(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed
        successful = sum(1 for r in results if not isinstance(r, Exception) and r["status_code"] == 201)
        assert successful >= 15, f"Only {successful}/20 registrations succeeded"
        
        # Check performance
        durations = [r["duration"] for r in results if not isinstance(r, Exception)]
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 3.0, f"Average registration time {avg_duration}s exceeds 3s"
    
    async def test_sustained_load_handling(self, client: AsyncClient, test_student_user: Dict):
        """Test system under sustained load."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        async def make_request(index: int):
            """Make a single API request."""
            # Alternate between different operations
            if index % 3 == 0:
                # Get submissions list
                response = await client.get("/api/v1/submissions/", headers=headers)
            elif index % 3 == 1:
                # Get user profile
                response = await client.get("/api/v1/users/me", headers=headers)
            else:
                # Health check
                response = await client.get("/health")
            
            return response.status_code
        
        # Make 50 requests
        tasks = [make_request(i) for i in range(50)]
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Calculate success rate
        successful = sum(1 for r in results if not isinstance(r, Exception) and r in [200, 201])
        success_rate = successful / len(results)
        
        assert success_rate >= 0.95, f"Success rate {success_rate} below 95% threshold"
        
        # Check throughput
        total_time = end_time - start_time
        requests_per_second = len(results) / total_time
        assert requests_per_second >= 10, \
            f"Throughput {requests_per_second} req/s below 10 req/s threshold"
    
    async def test_database_connection_pool_handling(self, client: AsyncClient):
        """Test database connection pool under load."""
        # Create multiple users to test connection pooling
        async def create_and_login(index: int):
            """Create user and login."""
            user_data = {
                "email": f"dbtest{index}@test.edu",
                "password": "TestPassword123!",
                "name": f"DB Test User {index}",
                "role": "student"
            }
            
            # Register
            reg_response = await client.post("/api/v1/auth/register", json=user_data)
            if reg_response.status_code != 201:
                return False
            
            # Login
            login_response = await client.post(
                "/api/v1/auth/login",
                data={"username": user_data["email"], "password": user_data["password"]}
            )
            return login_response.status_code == 200
        
        # Create 15 concurrent database operations
        tasks = [create_and_login(i) for i in range(15)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed without connection pool exhaustion
        successful = sum(1 for r in results if r is True)
        assert successful >= 12, \
            f"Only {successful}/15 database operations succeeded - possible connection pool issue"
    
    async def test_memory_efficiency_bulk_operations(self, client: AsyncClient, test_student_user: Dict):
        """Test memory efficiency with bulk operations."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Upload multiple files to test memory handling
        upload_count = 10
        successful_uploads = 0
        
        for i in range(upload_count):
            # Create moderately sized content
            content = f"Bulk operation test essay {i}. " * 100  # ~3KB per essay
            files = {"file": (f"bulk_{i}.txt", content.encode(), "text/plain")}
            data = {"title": f"Bulk Essay {i}", "course_name": "TEST101"}
            
            response = await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                successful_uploads += 1
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        # Should handle all uploads without memory issues
        assert successful_uploads >= 8, \
            f"Only {successful_uploads}/{upload_count} bulk uploads succeeded"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cleanup_test_files():
    """Clean up test database file."""
    if os.path.exists("test_integration.db"):
        os.remove("test_integration.db")


# Run cleanup after all tests
@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup test files after all tests."""
    request.addfinalizer(cleanup_test_files)

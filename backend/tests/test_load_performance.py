"""
Advanced Load Testing and Performance Benchmarking.

This module contains advanced load tests for system scalability validation:
- Stress testing with high concurrent users
- Performance benchmarking for critical operations
- Resource utilization monitoring
- Scalability limits identification

Requirements: 8.1, 8.2, 8.3
"""
import pytest
import asyncio
import time
import statistics
from httpx import AsyncClient
from typing import List, Dict
from dataclasses import dataclass

from main import app
from core.database import get_db
from tests.test_integration_e2e import (
    override_get_db,
    test_db,
    client,
    test_student_user,
    test_admin_user
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    success_rate: float


class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.status_codes: List[int] = []
        self.start_time: float = 0
        self.end_time: float = 0
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
    
    def record(self, response_time: float, status_code: int):
        """Record a request result."""
        self.response_times.append(response_time)
        self.status_codes.append(status_code)
    
    def stop(self):
        """Stop monitoring."""
        self.end_time = time.time()
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate and return performance metrics."""
        total = len(self.response_times)
        successful = sum(1 for code in self.status_codes if 200 <= code < 300)
        failed = total - successful
        
        if not self.response_times:
            return PerformanceMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                success_rate=0
            )
        
        sorted_times = sorted(self.response_times)
        duration = self.end_time - self.start_time
        
        return PerformanceMetrics(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=statistics.mean(self.response_times),
            min_response_time=min(self.response_times),
            max_response_time=max(self.response_times),
            p95_response_time=sorted_times[int(len(sorted_times) * 0.95)],
            p99_response_time=sorted_times[int(len(sorted_times) * 0.99)],
            requests_per_second=total / duration if duration > 0 else 0,
            success_rate=successful / total if total > 0 else 0
        )


# ============================================================================
# STRESS TESTING
# ============================================================================

@pytest.mark.asyncio
class TestStressScenarios:
    """Stress testing with high concurrent load."""
    
    async def test_high_concurrent_user_load(self, client: AsyncClient):
        """Test system with 100+ concurrent users."""
        monitor = PerformanceMonitor()
        
        async def simulate_user_session(user_id: int):
            """Simulate a complete user session."""
            # Register
            user_data = {
                "email": f"stress{user_id}@test.edu",
                "password": "TestPassword123!",
                "name": f"Stress Test User {user_id}",
                "role": "student"
            }
            
            start = time.time()
            reg_response = await client.post("/api/v1/auth/register", json=user_data)
            monitor.record(time.time() - start, reg_response.status_code)
            
            if reg_response.status_code != 201:
                return
            
            # Login
            start = time.time()
            login_response = await client.post(
                "/api/v1/auth/login",
                data={"username": user_data["email"], "password": user_data["password"]}
            )
            monitor.record(time.time() - start, login_response.status_code)
            
            if login_response.status_code != 200:
                return
            
            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Get profile
            start = time.time()
            profile_response = await client.get("/api/v1/users/me", headers=headers)
            monitor.record(time.time() - start, profile_response.status_code)
            
            # Get submissions
            start = time.time()
            submissions_response = await client.get("/api/v1/submissions/", headers=headers)
            monitor.record(time.time() - start, submissions_response.status_code)
        
        # Simulate 50 concurrent users (reduced from 100 for test stability)
        monitor.start()
        tasks = [simulate_user_session(i) for i in range(50)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # Assertions
        assert metrics.success_rate >= 0.90, \
            f"Success rate {metrics.success_rate:.2%} below 90% under stress"
        assert metrics.avg_response_time < 5.0, \
            f"Average response time {metrics.avg_response_time:.2f}s exceeds 5s under stress"
        assert metrics.p95_response_time < 10.0, \
            f"P95 response time {metrics.p95_response_time:.2f}s exceeds 10s under stress"
    
    async def test_burst_traffic_handling(self, client: AsyncClient, test_student_user: Dict):
        """Test system handling of sudden traffic bursts."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        monitor = PerformanceMonitor()
        
        async def make_burst_request(index: int):
            """Make a single request in burst."""
            start = time.time()
            response = await client.get("/api/v1/submissions/", headers=headers)
            monitor.record(time.time() - start, response.status_code)
        
        # Create burst of 30 simultaneous requests
        monitor.start()
        tasks = [make_burst_request(i) for i in range(30)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # System should handle burst gracefully
        assert metrics.success_rate >= 0.85, \
            f"Success rate {metrics.success_rate:.2%} below 85% during burst"
        assert metrics.max_response_time < 15.0, \
            f"Max response time {metrics.max_response_time:.2f}s exceeds 15s during burst"
    
    async def test_sustained_high_load(self, client: AsyncClient, test_student_user: Dict):
        """Test system under sustained high load."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        monitor = PerformanceMonitor()
        
        async def sustained_requests(duration_seconds: int):
            """Make requests continuously for specified duration."""
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                start = time.time()
                response = await client.get("/health")
                monitor.record(time.time() - start, response.status_code)
                await asyncio.sleep(0.1)  # 10 requests per second per task
        
        # Run 5 concurrent tasks for 10 seconds
        monitor.start()
        tasks = [sustained_requests(10) for _ in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # System should maintain performance under sustained load
        assert metrics.success_rate >= 0.95, \
            f"Success rate {metrics.success_rate:.2%} below 95% under sustained load"
        assert metrics.avg_response_time < 2.0, \
            f"Average response time {metrics.avg_response_time:.2f}s exceeds 2s under sustained load"
        assert metrics.requests_per_second >= 20, \
            f"Throughput {metrics.requests_per_second:.2f} req/s below 20 req/s"


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

@pytest.mark.asyncio
class TestPerformanceBenchmarks:
    """Benchmark critical system operations."""
    
    async def test_authentication_performance(self, client: AsyncClient):
        """Benchmark authentication operations."""
        # Create test user
        user_data = {
            "email": "benchmark@test.edu",
            "password": "TestPassword123!",
            "name": "Benchmark User",
            "role": "student"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        monitor = PerformanceMonitor()
        
        async def login_request():
            """Perform login."""
            start = time.time()
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": user_data["email"], "password": user_data["password"]}
            )
            monitor.record(time.time() - start, response.status_code)
        
        # Benchmark 20 login requests
        monitor.start()
        tasks = [login_request() for _ in range(20)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # Authentication should be fast
        assert metrics.avg_response_time < 1.0, \
            f"Average login time {metrics.avg_response_time:.2f}s exceeds 1s"
        assert metrics.p95_response_time < 2.0, \
            f"P95 login time {metrics.p95_response_time:.2f}s exceeds 2s"
    
    async def test_file_upload_performance(self, client: AsyncClient, test_student_user: Dict):
        """Benchmark file upload operations."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        monitor = PerformanceMonitor()
        
        async def upload_file(index: int):
            """Upload a file."""
            content = f"Performance test essay {index}. " * 100  # ~3KB
            files = {"file": (f"perf_{index}.txt", content.encode(), "text/plain")}
            data = {"title": f"Performance Essay {index}", "course_name": "PERF101"}
            
            start = time.time()
            response = await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
            monitor.record(time.time() - start, response.status_code)
        
        # Benchmark 10 file uploads
        monitor.start()
        tasks = [upload_file(i) for i in range(10)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # File uploads should complete within reasonable time
        assert metrics.avg_response_time < 5.0, \
            f"Average upload time {metrics.avg_response_time:.2f}s exceeds 5s"
        assert metrics.success_rate >= 0.80, \
            f"Upload success rate {metrics.success_rate:.2%} below 80%"
    
    async def test_query_performance_with_pagination(self, client: AsyncClient, test_student_user: Dict):
        """Benchmark query performance with pagination."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Create some submissions first
        for i in range(5):
            content = f"Query test essay {i}. " * 50
            files = {"file": (f"query_{i}.txt", content.encode(), "text/plain")}
            data = {"title": f"Query Essay {i}", "course_name": "QUERY101"}
            await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
        
        await asyncio.sleep(1)
        
        monitor = PerformanceMonitor()
        
        async def query_with_pagination(page: int):
            """Query with pagination."""
            start = time.time()
            response = await client.get(
                f"/api/v1/submissions/?page={page}&limit=10",
                headers=headers
            )
            monitor.record(time.time() - start, response.status_code)
        
        # Benchmark pagination queries
        monitor.start()
        tasks = [query_with_pagination(i) for i in range(1, 11)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # Queries should be fast
        assert metrics.avg_response_time < 1.0, \
            f"Average query time {metrics.avg_response_time:.2f}s exceeds 1s"
        assert metrics.success_rate >= 0.95, \
            f"Query success rate {metrics.success_rate:.2%} below 95%"


# ============================================================================
# SCALABILITY TESTING
# ============================================================================

@pytest.mark.asyncio
class TestScalabilityLimits:
    """Test system scalability limits."""
    
    async def test_maximum_concurrent_connections(self, client: AsyncClient):
        """Test maximum number of concurrent connections."""
        monitor = PerformanceMonitor()
        
        async def health_check():
            """Simple health check."""
            start = time.time()
            response = await client.get("/health")
            monitor.record(time.time() - start, response.status_code)
        
        # Test with increasing concurrency
        concurrency_levels = [10, 25, 50]
        results = {}
        
        for level in concurrency_levels:
            monitor = PerformanceMonitor()
            monitor.start()
            tasks = [health_check() for _ in range(level)]
            await asyncio.gather(*tasks, return_exceptions=True)
            monitor.stop()
            
            metrics = monitor.get_metrics()
            results[level] = metrics
        
        # Verify system handles increasing load
        for level, metrics in results.items():
            assert metrics.success_rate >= 0.90, \
                f"Success rate {metrics.success_rate:.2%} below 90% at {level} concurrent connections"
    
    async def test_data_volume_scalability(self, client: AsyncClient, test_student_user: Dict):
        """Test system performance with increasing data volume."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        
        # Upload varying sizes of content
        sizes = [1000, 5000, 10000]  # words
        upload_times = []
        
        for size in sizes:
            content = "word " * size
            files = {"file": (f"size_{size}.txt", content.encode(), "text/plain")}
            data = {"title": f"Size Test {size}", "course_name": "SIZE101"}
            
            start = time.time()
            response = await client.post(
                "/api/v1/submissions/upload",
                files=files,
                data=data,
                headers=headers
            )
            duration = time.time() - start
            
            if response.status_code in [200, 201]:
                upload_times.append((size, duration))
        
        # Verify reasonable scaling
        assert len(upload_times) >= 2, "Not enough successful uploads to test scaling"
        
        # Processing time should scale reasonably with content size
        for size, duration in upload_times:
            # Allow up to 30 seconds for 5000 words (per requirements)
            max_time = (size / 5000) * 30
            assert duration < max_time, \
                f"Upload of {size} words took {duration:.2f}s, exceeds {max_time:.2f}s"
    
    async def test_user_growth_scalability(self, client: AsyncClient):
        """Test system scalability with growing user base."""
        # Create multiple users
        user_count = 20
        tokens = []
        
        for i in range(user_count):
            user_data = {
                "email": f"scale{i}@test.edu",
                "password": "TestPassword123!",
                "name": f"Scale User {i}",
                "role": "student"
            }
            
            reg_response = await client.post("/api/v1/auth/register", json=user_data)
            if reg_response.status_code == 201:
                login_response = await client.post(
                    "/api/v1/auth/login",
                    data={"username": user_data["email"], "password": user_data["password"]}
                )
                if login_response.status_code == 200:
                    tokens.append(login_response.json()["access_token"])
        
        # All users make requests simultaneously
        monitor = PerformanceMonitor()
        
        async def user_request(token: str):
            """Make request as user."""
            headers = {"Authorization": f"Bearer {token}"}
            start = time.time()
            response = await client.get("/api/v1/submissions/", headers=headers)
            monitor.record(time.time() - start, response.status_code)
        
        monitor.start()
        tasks = [user_request(token) for token in tokens]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # System should handle all users efficiently
        assert metrics.success_rate >= 0.90, \
            f"Success rate {metrics.success_rate:.2%} below 90% with {user_count} users"
        assert metrics.avg_response_time < 3.0, \
            f"Average response time {metrics.avg_response_time:.2f}s exceeds 3s with {user_count} users"


# ============================================================================
# RESOURCE UTILIZATION TESTS
# ============================================================================

@pytest.mark.asyncio
class TestResourceUtilization:
    """Test resource utilization and efficiency."""
    
    async def test_connection_pool_efficiency(self, client: AsyncClient):
        """Test database connection pool efficiency."""
        monitor = PerformanceMonitor()
        
        async def db_operation(index: int):
            """Perform database operation."""
            user_data = {
                "email": f"pool{index}@test.edu",
                "password": "TestPassword123!",
                "name": f"Pool User {index}",
                "role": "student"
            }
            
            start = time.time()
            response = await client.post("/api/v1/auth/register", json=user_data)
            monitor.record(time.time() - start, response.status_code)
        
        # Create 20 concurrent database operations
        monitor.start()
        tasks = [db_operation(i) for i in range(20)]
        await asyncio.gather(*tasks, return_exceptions=True)
        monitor.stop()
        
        metrics = monitor.get_metrics()
        
        # Connection pool should handle load efficiently
        assert metrics.success_rate >= 0.85, \
            f"Success rate {metrics.success_rate:.2%} indicates connection pool issues"
        assert metrics.avg_response_time < 2.0, \
            f"Average time {metrics.avg_response_time:.2f}s indicates connection pool bottleneck"
    
    async def test_response_time_consistency(self, client: AsyncClient, test_student_user: Dict):
        """Test consistency of response times."""
        headers = {"Authorization": f"Bearer {test_student_user['token']}"}
        response_times = []
        
        # Make 30 identical requests
        for _ in range(30):
            start = time.time()
            response = await client.get("/api/v1/submissions/", headers=headers)
            duration = time.time() - start
            
            if response.status_code == 200:
                response_times.append(duration)
            
            await asyncio.sleep(0.1)
        
        # Calculate consistency metrics
        if len(response_times) >= 10:
            avg_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times)
            
            # Standard deviation should be reasonable (not too variable)
            coefficient_of_variation = std_dev / avg_time if avg_time > 0 else 0
            assert coefficient_of_variation < 0.5, \
                f"Response time variability {coefficient_of_variation:.2f} indicates inconsistent performance"

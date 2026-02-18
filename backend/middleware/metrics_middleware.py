"""
Metrics middleware for collecting application performance metrics
"""
import time
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of active users'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

SUBMISSION_COUNT = Counter(
    'submissions_total',
    'Total number of submissions processed',
    ['status']
)

VERIFICATION_DURATION = Histogram(
    'verification_duration_seconds',
    'Time taken for verification processing',
    ['verification_type']
)

BLOCKCHAIN_OPERATIONS = Counter(
    'blockchain_operations_total',
    'Total blockchain operations',
    ['operation', 'status']
)

class MetricsMiddleware:
    """Middleware for collecting HTTP request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            await self.app(scope, receive, send)
            return
        
        method = request.method
        path = request.url.path
        
        # Normalize path for metrics (remove IDs)
        endpoint = self._normalize_path(path)
        
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            # Process request
            response = Response()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    response.status_code = message["status"]
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
            
            # Record metrics
            duration = time.time() - start_time
            status = str(response.status_code)
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status="500"
            ).inc()
            
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            logger.error(f"Request failed: {e}")
            raise
        
        finally:
            ACTIVE_REQUESTS.dec()
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path by removing IDs and parameters"""
        # Replace UUIDs and numeric IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path

def record_submission_metric(status: str):
    """Record submission processing metric"""
    SUBMISSION_COUNT.labels(status=status).inc()

def record_verification_metric(verification_type: str, duration: float):
    """Record verification processing metric"""
    VERIFICATION_DURATION.labels(verification_type=verification_type).observe(duration)

def record_blockchain_metric(operation: str, status: str):
    """Record blockchain operation metric"""
    BLOCKCHAIN_OPERATIONS.labels(operation=operation, status=status).inc()

def update_active_users(count: int):
    """Update active users gauge"""
    ACTIVE_USERS.set(count)

def update_database_connections(count: int):
    """Update database connections gauge"""
    DATABASE_CONNECTIONS.set(count)

async def metrics_endpoint():
    """Endpoint to expose Prometheus metrics"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
"""
Security middleware for request monitoring and protection.
"""
import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

from core.config import settings
from core.database import get_db
from services.security_service import security_service
from services.audit_service import audit_service
from core.redis import cache

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security monitoring and protection."""
    
    def __init__(self, app, enable_intrusion_detection: bool = True):
        super().__init__(app)
        self.enable_intrusion_detection = enable_intrusion_detection
        
        # Paths that don't require security monitoring
        self.excluded_paths = {
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json"
        }
        
        # High-risk paths that require extra monitoring
        self.high_risk_paths = {
            "/api/v1/auth/login", "/api/v1/auth/register", 
            "/api/v1/admin/", "/api/v1/submissions/upload"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        start_time = time.time()
        
        # Skip security checks for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        request_path = request.url.path
        
        try:
            # Rate limiting check
            if settings.RATE_LIMIT_ENABLED:
                rate_limit_result = await self._check_rate_limit(client_ip, request_path)
                if rate_limit_result["blocked"]:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "detail": "Rate limit exceeded",
                            "retry_after": rate_limit_result["retry_after"]
                        }
                    )
            
            # Intrusion detection
            if self.enable_intrusion_detection and settings.ENABLE_INTRUSION_DETECTION:
                async with get_db() as db:
                    intrusion_result = await security_service.detect_intrusion_attempt(
                        db=db,
                        ip_address=client_ip,
                        user_agent=user_agent,
                        request_path=request_path
                    )
                    
                    # Block high-risk requests
                    if intrusion_result["risk_score"] >= 80:
                        await self._log_blocked_request(
                            db, client_ip, user_agent, request_path, intrusion_result
                        )
                        return JSONResponse(
                            status_code=status.HTTP_403_FORBIDDEN,
                            content={"detail": "Request blocked for security reasons"}
                        )
            
            # Process request
            response = await call_next(request)
            
            # Log request if it's high-risk or resulted in error
            if (any(request_path.startswith(path) for path in self.high_risk_paths) or 
                response.status_code >= 400):
                await self._log_request(
                    client_ip, user_agent, request_path, 
                    response.status_code, time.time() - start_time
                )
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # Don't block requests due to middleware errors
            response = await call_next(request)
            self._add_security_headers(response)
            return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"
    
    async def _check_rate_limit(self, client_ip: str, request_path: str) -> dict:
        """Check rate limiting for client IP."""
        try:
            # Different limits for different path types
            if any(request_path.startswith(path) for path in self.high_risk_paths):
                limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE // 2  # Stricter for high-risk
            else:
                limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
            
            # Check current request count
            key = f"rate_limit:{client_ip}"
            current_count = await cache.get(key) or 0
            if isinstance(current_count, str):
                current_count = int(current_count)
            
            if current_count >= limit:
                return {"blocked": True, "retry_after": 60}
            
            # Increment counter
            await cache.set(key, current_count + 1, expire=60)
            
            return {"blocked": False, "current_count": current_count + 1}
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return {"blocked": False, "current_count": 0}
    
    async def _log_blocked_request(
        self, 
        db, 
        client_ip: str, 
        user_agent: str, 
        request_path: str, 
        intrusion_result: dict
    ) -> None:
        """Log blocked request for security monitoring."""
        try:
            await audit_service.log_event(
                db=db,
                event_type="REQUEST_BLOCKED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Request blocked from {client_ip} to {request_path}",
                ip_address=client_ip,
                user_agent=user_agent,
                metadata={
                    "request_path": request_path,
                    "risk_score": intrusion_result["risk_score"],
                    "threats_detected": intrusion_result["threats_detected"]
                },
                risk_level=audit_service.RiskLevel.HIGH
            )
        except Exception as e:
            logger.error(f"Error logging blocked request: {e}")
    
    async def _log_request(
        self, 
        client_ip: str, 
        user_agent: str, 
        request_path: str, 
        status_code: int, 
        response_time: float
    ) -> None:
        """Log high-risk or error requests."""
        try:
            # Only log to cache for performance (detailed logs go to audit service)
            log_data = {
                "ip": client_ip,
                "path": request_path,
                "status": status_code,
                "response_time": response_time,
                "timestamp": time.time()
            }
            
            # Store in Redis for real-time monitoring
            await cache.lpush("security_requests", str(log_data))
            await cache.ltrim("security_requests", 0, 999)  # Keep last 1000 requests
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        try:
            # Content Security Policy
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none';"
            )
            
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            # HTTPS enforcement (if enabled)
            if settings.FORCE_HTTPS:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            # Remove server information
            response.headers.pop("server", None)
            
        except Exception as e:
            logger.error(f"Error adding security headers: {e}")


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking request patterns and anomalies."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request patterns for anomaly detection."""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            # Track request count per IP per minute
            minute_key = f"requests_per_minute:{client_ip}:{int(time.time() // 60)}"
            await cache.incr(minute_key)
            await cache.expire(minute_key, 120)  # Keep for 2 minutes
            
            # Track unique paths per IP
            paths_key = f"unique_paths:{client_ip}"
            await cache.sadd(paths_key, request.url.path)
            await cache.expire(paths_key, 3600)  # Keep for 1 hour
            
            response = await call_next(request)
            
            # Track response times for anomaly detection
            response_time = time.time() - start_time
            if response_time > 5.0:  # Log slow requests
                await self._log_slow_request(client_ip, request.url.path, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request tracking error: {e}")
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _log_slow_request(self, client_ip: str, path: str, response_time: float) -> None:
        """Log slow requests for performance monitoring."""
        try:
            slow_request_data = {
                "ip": client_ip,
                "path": path,
                "response_time": response_time,
                "timestamp": time.time()
            }
            
            await cache.lpush("slow_requests", str(slow_request_data))
            await cache.ltrim("slow_requests", 0, 99)  # Keep last 100 slow requests
            
        except Exception as e:
            logger.error(f"Error logging slow request: {e}")


# Middleware factory functions
def create_security_middleware(enable_intrusion_detection: bool = True):
    """Create security middleware with configuration."""
    def middleware_factory(app):
        return SecurityMiddleware(app, enable_intrusion_detection)
    return middleware_factory


def create_request_tracking_middleware():
    """Create request tracking middleware."""
    def middleware_factory(app):
        return RequestTrackingMiddleware(app)
    return middleware_factory
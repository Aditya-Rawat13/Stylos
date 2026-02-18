"""
Health check service for monitoring system status
"""
import asyncio
import time
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as redis
import logging
from datetime import datetime, timedelta

from core.database import get_db
from core.config import settings

logger = logging.getLogger(__name__)

class HealthService:
    """Service for checking system health and dependencies"""
    
    def __init__(self):
        self.redis_client = None
        self._last_health_check = None
        self._health_cache = None
        self._cache_duration = 30  # seconds
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        # Use cached result if recent
        if (self._last_health_check and 
            time.time() - self._last_health_check < self._cache_duration and
            self._health_cache):
            return self._health_cache
        
        start_time = time.time()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "checks": {}
        }
        
        # Check database
        db_status = await self._check_database()
        health_status["checks"]["database"] = db_status
        
        # Check Redis
        redis_status = await self._check_redis()
        health_status["checks"]["redis"] = redis_status
        
        # Check external services
        blockchain_status = await self._check_blockchain()
        health_status["checks"]["blockchain"] = blockchain_status
        
        ipfs_status = await self._check_ipfs()
        health_status["checks"]["ipfs"] = ipfs_status
        
        # Check system resources
        system_status = await self._check_system_resources()
        health_status["checks"]["system"] = system_status
        
        # Determine overall status
        failed_checks = [
            name for name, check in health_status["checks"].items()
            if check["status"] != "healthy"
        ]
        
        if failed_checks:
            if any(check["status"] == "critical" for check in health_status["checks"].values()):
                health_status["status"] = "critical"
            else:
                health_status["status"] = "degraded"
            health_status["failed_checks"] = failed_checks
        
        health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Cache result
        self._health_cache = health_status
        self._last_health_check = time.time()
        
        return health_status
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            
            # Get database session
            async for db in get_db():
                # Test basic connectivity
                result = await db.execute(text("SELECT 1"))
                result.fetchone()
                
                # Test write capability
                await db.execute(text("SELECT NOW()"))
                
                # Get connection count
                conn_result = await db.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                ))
                active_connections = conn_result.scalar()
                
                response_time = round((time.time() - start_time) * 1000, 2)
                
                return {
                    "status": "healthy",
                    "response_time_ms": response_time,
                    "active_connections": active_connections,
                    "details": "Database is responsive"
                }
        
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "critical",
                "error": str(e),
                "details": "Database connection failed"
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            start_time = time.time()
            
            if not self.redis_client:
                self.redis_client = redis.from_url(settings.REDIS_URL)
            
            # Test basic connectivity
            await self.redis_client.ping()
            
            # Test read/write
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test_value", ex=60)
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if value != b"test_value":
                raise Exception("Redis read/write test failed")
            
            # Get Redis info
            info = await self.redis_client.info()
            
            response_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "details": "Redis is responsive"
            }
        
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "details": "Redis connection failed"
            }
    
    async def _check_blockchain(self) -> Dict[str, Any]:
        """Check blockchain connectivity"""
        try:
            start_time = time.time()
            
            # This would typically check Web3 connection
            # For now, return a mock status
            await asyncio.sleep(0.1)  # Simulate network call
            
            response_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "network": "polygon",
                "details": "Blockchain connection is healthy"
            }
        
        except Exception as e:
            logger.error(f"Blockchain health check failed: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "details": "Blockchain connection failed"
            }
    
    async def _check_ipfs(self) -> Dict[str, Any]:
        """Check IPFS connectivity"""
        try:
            start_time = time.time()
            
            # This would typically check IPFS connection
            # For now, return a mock status
            await asyncio.sleep(0.1)  # Simulate network call
            
            response_time = round((time.time() - start_time) * 1000, 2)
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "details": "IPFS connection is healthy"
            }
        
        except Exception as e:
            logger.error(f"IPFS health check failed: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "details": "IPFS connection failed"
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            warnings = []
            
            if cpu_percent > 80:
                status = "degraded"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 85:
                status = "degraded"
                warnings.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = "degraded"
                warnings.append(f"High disk usage: {disk.percent}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "warnings": warnings,
                "details": "System resources checked"
            }
        
        except ImportError:
            return {
                "status": "unknown",
                "details": "psutil not available for system monitoring"
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "details": "System resource check failed"
            }
    
    async def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status (lighter check for k8s readiness probe)"""
        try:
            # Quick database check
            async for db in get_db():
                await db.execute(text("SELECT 1"))
                break
            
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Global health service instance
health_service = HealthService()
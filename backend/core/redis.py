"""
Redis configuration and utilities.
"""
import redis.asyncio as redis
import json
import logging
from typing import Any, Optional
from datetime import timedelta

from core.config import settings

logger = logging.getLogger(__name__)

# Redis connection pool
redis_pool = None


async def init_redis():
    """Initialize Redis connection pool."""
    global redis_pool
    try:
        redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20
        )
        # Test connection
        async with redis.Redis(connection_pool=redis_pool) as r:
            await r.ping()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        raise


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if redis_pool is None:
        await init_redis()
    return redis.Redis(connection_pool=redis_pool)


class CacheService:
    """Redis cache service."""
    
    def __init__(self):
        self.redis_client = None
    
    async def _get_client(self) -> redis.Redis:
        """Get Redis client instance."""
        if self.redis_client is None:
            self.redis_client = await get_redis()
        return self.redis_client
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[timedelta] = None
    ) -> bool:
        """Set a value in cache."""
        try:
            client = await self._get_client()
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            if expire:
                return await client.setex(key, expire, serialized_value)
            else:
                return await client.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            client = await self._get_client()
            value = await client.get(key)
            
            if value is None:
                return None
            
            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            client = await self._get_client()
            return bool(await client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            client = await self._get_client()
            return bool(await client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False


# Global cache service instance
cache = CacheService()
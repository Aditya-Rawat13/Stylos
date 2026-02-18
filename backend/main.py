"""
Production FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from core.config import settings
from core.database import init_db
from core.redis import init_redis
from api.v1.api import api_router
# from middleware.metrics_middleware import MetricsMiddleware, metrics_endpoint
from services.health_service import health_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting up TrueAuthor production application...")
    await init_db()
    await init_redis()
    
    # Initialize verification service
    try:
        from services.verification_service import verification_service
        logger.info("Initializing verification service...")
        await verification_service.initialize()
        logger.info("Verification service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize verification service: {e}")
        logger.warning("Application will continue but verification may not work properly")
    
    yield
    # Shutdown
    logger.info("Shutting down TrueAuthor production application...")


app = FastAPI(
    title="TrueAuthor - Academic Writing Verification System",
    description="Production API for academic writing verification and blockchain attestation",
    version="1.0.0",
    lifespan=lifespan
)

# Add metrics middleware
# app.add_middleware(MetricsMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    return await health_service.get_health_status()


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint for Kubernetes."""
    return await health_service.get_readiness_status()


# @app.get("/metrics")
# async def get_metrics():
#     """Prometheus metrics endpoint."""
#     return await metrics_endpoint()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
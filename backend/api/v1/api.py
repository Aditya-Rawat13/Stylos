"""
API v1 router configuration.
"""
from fastapi import APIRouter

from api.v1.endpoints import auth, users, submissions, admin, blockchain, similarity, security, lms_integration
from .endpoints import verification

api_router = APIRouter()


@api_router.get("/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "ok", "message": "Project Stylos API is running"}


@api_router.get("/test-auth")
async def test_auth():
    """Test endpoint to get authentication token."""
    return {
        "message": "Use POST /api/v1/auth/create-test-user to get a test token",
        "instructions": "This will create a test user and return authentication tokens"
    }

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
# Also include users router under /profile for frontend compatibility
api_router.include_router(users.router, prefix="/profile", tags=["profile"])
api_router.include_router(submissions.router, prefix="/submissions", tags=["submissions"])
api_router.include_router(verification.router, prefix="/verification", tags=["verification"])
api_router.include_router(blockchain.router, prefix="/blockchain", tags=["blockchain"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(similarity.router, prefix="/similarity", tags=["similarity"])
api_router.include_router(security.router, prefix="/security", tags=["security"])
api_router.include_router(lms_integration.router, prefix="/lms", tags=["lms-integration"])
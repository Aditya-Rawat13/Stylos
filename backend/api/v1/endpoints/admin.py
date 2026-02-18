"""
Admin endpoints.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard")
async def get_admin_dashboard():
    """Get admin dashboard data."""
    return {"message": "Get admin dashboard endpoint - to be implemented"}


@router.get("/submissions/flagged")
async def get_flagged_submissions():
    """Get flagged submissions for review."""
    return {"message": "Get flagged submissions endpoint - to be implemented"}


@router.post("/submissions/{submission_id}/approve")
async def approve_submission():
    """Approve a flagged submission."""
    return {"message": "Approve submission endpoint - to be implemented"}


@router.post("/submissions/{submission_id}/reject")
async def reject_submission():
    """Reject a flagged submission."""
    return {"message": "Reject submission endpoint - to be implemented"}
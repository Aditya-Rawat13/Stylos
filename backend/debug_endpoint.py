#!/usr/bin/env python3
"""
Debug endpoint to add to the backend for testing.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.auth import get_current_active_user
from models.user import User
from schemas.user import UserProfileResponse, WritingProfileResponse
from services.user_service import user_service

debug_router = APIRouter()

@debug_router.get("/debug-profile")
async def debug_profile_endpoint(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Debug version of profile endpoint."""
    try:
        # Step 1: Check current user
        result = {"step": 1, "current_user_id": current_user.id, "current_user_email": current_user.email}
        
        # Step 2: Call user service
        user = await user_service.get_user_profile(db, current_user.id)
        result["step"] = 2
        result["user_service_result"] = user.email if user else None
        
        if not user:
            result["error"] = "User service returned None"
            return result
        
        # Step 3: Check writing profile
        result["step"] = 3
        result["has_writing_profile"] = user.writing_profile is not None
        
        # Step 4: Convert writing profile
        writing_profile_response = None
        if user.writing_profile:
            try:
                writing_profile_response = WritingProfileResponse.from_orm(user.writing_profile)
                result["writing_profile_conversion"] = "success"
            except Exception as e:
                result["writing_profile_conversion"] = f"failed: {str(e)}"
                return result
        
        result["step"] = 4
        
        # Step 5: Create response
        try:
            response = UserProfileResponse(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                is_active=user.is_active,
                is_verified=user.is_verified,
                institution_id=user.institution_id,
                student_id=user.student_id,
                created_at=user.created_at,
                last_login=user.last_login,
                writing_profile=writing_profile_response
            )
            result["step"] = 5
            result["success"] = True
            result["response_email"] = response.email
            return result
            
        except Exception as e:
            result["step"] = 5
            result["response_creation_error"] = str(e)
            return result
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "step": result.get("step", 0)
        }

# To add this to the main app, add this line to api/v1/api.py:
# api_router.include_router(debug_router, prefix="/debug", tags=["debug"])
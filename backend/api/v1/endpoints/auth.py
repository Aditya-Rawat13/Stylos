"""
Authentication endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.auth import get_current_active_user
from models.user import User
from schemas.auth import (
    UserLogin, UserRegister, TokenResponse, RefreshTokenRequest,
    PasswordResetRequest, PasswordResetConfirm, EmailVerificationRequest,
    AuthResponse, UserResponse, ChangePasswordRequest
)
from services.auth_service import auth_service

router = APIRouter()


@router.post("/create-test-user", response_model=AuthResponse)
async def create_test_user(
    db: AsyncSession = Depends(get_db)
):
    """Create a test user for development (remove in production)."""
    from models.user import UserRole
    from sqlalchemy import select
    
    # Check if test user already exists
    result = await db.execute(select(User).where(User.email == "test@stylos.dev"))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        # Ensure existing test user is verified
        if not existing_user.is_verified:
            existing_user.is_verified = True
            await db.commit()
            await db.refresh(existing_user)
        
        # Login existing test user
        login_data = UserLogin(email="test@stylos.dev", password="TestPassword123")
        user, tokens = await auth_service.login_user(db, login_data, None)
        return AuthResponse(
            user=UserResponse.from_orm(user),
            tokens=tokens
        )
    
    # Create new test user
    test_user_data = UserRegister(
        email="test@stylos.dev",
        password="TestPassword123",
        confirm_password="TestPassword123",
        full_name="Test User"
    )
    
    user = await auth_service.register_user(db, test_user_data, None)
    
    # Mark test user as verified for development
    user.is_verified = True
    await db.commit()
    await db.refresh(user)
    
    # Auto-login the test user
    login_data = UserLogin(email="test@stylos.dev", password="TestPassword123")
    user, tokens = await auth_service.login_user(db, login_data, None)
    
    return AuthResponse(
        user=UserResponse.from_orm(user),
        tokens=tokens
    )


@router.post("/verify-user/{email}")
async def verify_user_dev(
    email: str,
    db: AsyncSession = Depends(get_db)
):
    """Verify a user for development (remove in production)."""
    from sqlalchemy import select, update
    
    # Find user by email
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify the user
    user.is_verified = True
    await db.commit()
    await db.refresh(user)
    
    return {"message": f"User {email} has been verified", "user_id": user.id}


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    user = await auth_service.register_user(db, user_data, request)
    return UserResponse.from_orm(user)


@router.post("/login", response_model=AuthResponse)
async def login(
    login_data: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """User login endpoint."""
    user, tokens = await auth_service.login_user(db, login_data, request)
    
    return AuthResponse(
        user=UserResponse.from_orm(user),
        tokens=tokens
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """Refresh JWT token endpoint."""
    tokens = await auth_service.refresh_access_token(db, refresh_data.refresh_token)
    return tokens


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """User logout endpoint."""
    await auth_service.logout_user(current_user.id)


@router.post("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    verification_data: EmailVerificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify user email address."""
    success = await auth_service.verify_email(db, verification_data.token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    return {"message": "Email verified successfully"}


@router.post("/request-password-reset", status_code=status.HTTP_200_OK)
async def request_password_reset(
    reset_data: PasswordResetRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset."""
    await auth_service.generate_password_reset_token(db, reset_data.email, request)
    
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    reset_data: PasswordResetConfirm,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Reset password using reset token."""
    success = await auth_service.reset_password(db, reset_data.token, reset_data.new_password, request)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    return {"message": "Password reset successfully"}


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    password_data: ChangePasswordRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password."""
    success = await auth_service.change_password(
        db, current_user.id, password_data.current_password, 
        password_data.new_password, request
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to change password"
        )
    
    return {"message": "Password changed successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return UserResponse.from_orm(current_user)
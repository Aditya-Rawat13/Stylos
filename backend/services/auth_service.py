"""
Authentication service for user management and token handling.
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from fastapi import HTTPException, status, Request
import redis.asyncio as redis
import asyncio

from core.config import settings
from core.security import security
from core.redis import get_redis
from models.user import User, UserRole, WritingProfile
from schemas.auth import UserRegister, UserLogin, TokenResponse
from schemas.lms import LTILaunchRequest
from services.email_service import email_service
from services.security_service import security_service
from services.audit_service import audit_service


class AuthService:
    """Authentication service class."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            self.redis_client = await get_redis()
        return self.redis_client
    
    async def register_user(
        self, 
        db: AsyncSession, 
        user_data: UserRegister,
        request: Optional[Request] = None
    ) -> User:
        """Register a new user."""
        # Check if user already exists
        result = await db.execute(select(User).where(User.email == user_data.email))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate password strength
        password_validation = await security_service.validate_password_strength(user_data.password)
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password does not meet requirements: {', '.join(password_validation['feedback'])}"
            )
        
        # Create new user
        hashed_password = security.hash_password(user_data.password)
        
        new_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=UserRole.STUDENT,  # Default role
            institution_id=user_data.institution_id,
            student_id=user_data.student_id,
            is_active=True,
            is_verified=False  # Require email verification
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Create empty writing profile
        writing_profile = WritingProfile(
            user_id=new_user.id,
            is_initialized=False
        )
        db.add(writing_profile)
        await db.commit()
        
        # Generate and send verification email
        verification_token = await self.generate_verification_token(new_user.id)
        
        # Send verification email asynchronously
        asyncio.create_task(
            email_service.send_verification_email(
                new_user.email,
                new_user.full_name,
                verification_token
            )
        )
        
        # Log registration event
        ip_address = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None
        
        await audit_service.log_event(
            db=db,
            event_type="USER_REGISTRATION",
            event_category=audit_service.Category.AUTH,
            description=f"New user registered: {new_user.email}",
            user_id=new_user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_level=audit_service.RiskLevel.LOW
        )
        
        return new_user
    
    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not security.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        await db.execute(
            update(User)
            .where(User.id == user.id)
            .values(last_login=datetime.utcnow())
        )
        await db.commit()
        
        return user
    
    async def login_user(
        self, 
        db: AsyncSession, 
        login_data: UserLogin,
        request: Optional[Request] = None
    ) -> Tuple[User, TokenResponse]:
        """Login user and return tokens."""
        ip_address = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None
        
        # Check rate limiting
        is_limited, current_count, limit = await security_service.check_rate_limit(
            ip_address or "unknown", "login"
        )
        
        if is_limited:
            await audit_service.log_event(
                db=db,
                event_type=audit_service.EventType.RATE_LIMIT_EXCEEDED,
                event_category=audit_service.Category.SYSTEM,
                description=f"Rate limit exceeded for IP {ip_address}",
                ip_address=ip_address,
                user_agent=user_agent,
                risk_level=audit_service.RiskLevel.HIGH
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later."
            )
        
        user = await self.authenticate_user(db, login_data.email, login_data.password)
        
        # Track login attempt
        login_result = await security_service.track_login_attempt(
            db, login_data.email, ip_address, user_agent, success=bool(user)
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check account status
        await security_service.require_account_verification(user)
        
        # Create tokens
        access_token = security.create_access_token(data={"sub": str(user.id)})
        refresh_token = security.create_refresh_token(data={"sub": str(user.id)})
        
        # Store refresh token in Redis
        redis_client = await self.get_redis()
        await redis_client.setex(
            f"refresh_token:{user.id}",
            timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            refresh_token
        )
        
        tokens = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
        return user, tokens
    
    async def refresh_access_token(self, db: AsyncSession, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token."""
        # Verify refresh token
        payload = security.verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if refresh token exists in Redis
        redis_client = await self.get_redis()
        stored_token = await redis_client.get(f"refresh_token:{user_id}")
        
        if not stored_token or stored_token.decode() != refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Get user
        result = await db.execute(select(User).where(User.id == int(user_id)))
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        new_access_token = security.create_access_token(data={"sub": str(user.id)})
        new_refresh_token = security.create_refresh_token(data={"sub": str(user.id)})
        
        # Update refresh token in Redis
        await redis_client.setex(
            f"refresh_token:{user.id}",
            timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            new_refresh_token
        )
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    async def logout_user(self, user_id: int) -> None:
        """Logout user by invalidating refresh token."""
        redis_client = await self.get_redis()
        await redis_client.delete(f"refresh_token:{user_id}")
    
    async def generate_verification_token(self, user_id: int) -> str:
        """Generate email verification token."""
        token = security.generate_verification_token()
        
        # Store in Redis with 24-hour expiration
        redis_client = await self.get_redis()
        await redis_client.setex(
            f"email_verification:{token}",
            timedelta(hours=24),
            str(user_id)
        )
        
        return token
    
    async def verify_email(self, db: AsyncSession, token: str) -> bool:
        """Verify email using verification token."""
        redis_client = await self.get_redis()
        user_id_str = await redis_client.get(f"email_verification:{token}")
        
        if not user_id_str:
            return False
        
        user_id = int(user_id_str.decode())
        
        # Get user for welcome email
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            return False
        
        # Update user verification status
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(is_verified=True)
        )
        await db.commit()
        
        # Delete verification token
        await redis_client.delete(f"email_verification:{token}")
        
        # Send welcome email
        asyncio.create_task(
            email_service.send_welcome_email(user.email, user.full_name)
        )
        
        # Log verification event
        await audit_service.log_event(
            db=db,
            event_type=audit_service.EventType.EMAIL_VERIFICATION,
            event_category=audit_service.Category.AUTH,
            description=f"Email verified for user {user_id}",
            user_id=user_id,
            risk_level=audit_service.RiskLevel.LOW
        )
        
        return True
    
    async def generate_password_reset_token(
        self, 
        db: AsyncSession, 
        email: str,
        request: Optional[Request] = None
    ) -> Optional[str]:
        """Generate password reset token."""
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        # Always return success for security (don't reveal if email exists)
        if not user:
            # Log attempt for non-existent email
            ip_address = request.client.host if request else None
            await audit_service.log_event(
                db=db,
                event_type=audit_service.EventType.PASSWORD_RESET_REQUEST,
                event_category=audit_service.Category.AUTH,
                description=f"Password reset requested for non-existent email: {email}",
                ip_address=ip_address,
                metadata={"email": email},
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            return "dummy_token"  # Return dummy token for security
        
        token = security.generate_reset_token()
        
        # Store in Redis with 1-hour expiration
        redis_client = await self.get_redis()
        await redis_client.setex(
            f"password_reset:{token}",
            timedelta(hours=1),
            str(user.id)
        )
        
        # Send password reset email
        asyncio.create_task(
            email_service.send_password_reset_email(
                user.email, user.full_name, token
            )
        )
        
        # Log password reset request
        ip_address = request.client.host if request else None
        await audit_service.log_event(
            db=db,
            event_type=audit_service.EventType.PASSWORD_RESET_REQUEST,
            event_category=audit_service.Category.AUTH,
            description=f"Password reset requested for user {user.id}",
            user_id=user.id,
            ip_address=ip_address,
            risk_level=audit_service.RiskLevel.MEDIUM
        )
        
        return token
    
    async def reset_password(
        self, 
        db: AsyncSession, 
        token: str, 
        new_password: str,
        request: Optional[Request] = None
    ) -> bool:
        """Reset password using reset token."""
        redis_client = await self.get_redis()
        user_id_str = await redis_client.get(f"password_reset:{token}")
        
        if not user_id_str:
            return False
        
        user_id = int(user_id_str.decode())
        
        # Validate new password strength
        password_validation = await security_service.validate_password_strength(new_password)
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password does not meet requirements: {', '.join(password_validation['feedback'])}"
            )
        
        hashed_password = security.hash_password(new_password)
        
        # Update user password
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(hashed_password=hashed_password)
        )
        await db.commit()
        
        # Delete reset token
        await redis_client.delete(f"password_reset:{token}")
        
        # Invalidate all refresh tokens for this user
        await redis_client.delete(f"refresh_token:{user_id}")
        
        # Clear any account lockouts
        await redis_client.delete(f"lockout:{user_id}")
        await redis_client.delete(f"failed_logins:{user_id}")
        
        # Log password reset success
        ip_address = request.client.host if request else None
        await audit_service.log_event(
            db=db,
            event_type=audit_service.EventType.PASSWORD_RESET_SUCCESS,
            event_category=audit_service.Category.AUTH,
            description=f"Password reset completed for user {user_id}",
            user_id=user_id,
            ip_address=ip_address,
            risk_level=audit_service.RiskLevel.MEDIUM
        )
        
        return True
    
    async def change_password(
        self,
        db: AsyncSession,
        user_id: int,
        current_password: str,
        new_password: str,
        request: Optional[Request] = None
    ) -> bool:
        """Change user password with current password verification."""
        # Get user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not security.verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        password_validation = await security_service.validate_password_strength(new_password)
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password does not meet requirements: {', '.join(password_validation['feedback'])}"
            )
        
        # Check if new password is different from current
        if security.verify_password(new_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Update password
        hashed_password = security.hash_password(new_password)
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(hashed_password=hashed_password)
        )
        await db.commit()
        
        # Invalidate all refresh tokens for this user
        redis_client = await self.get_redis()
        await redis_client.delete(f"refresh_token:{user_id}")
        
        # Log password change
        ip_address = request.client.host if request else None
        await audit_service.log_password_change(
            db, user_id, ip_address, request.headers.get("user-agent") if request else None
        )
        
        # Send security notification email
        asyncio.create_task(
            email_service.send_security_alert(
                user.email,
                user.full_name,
                "Password Changed",
                "Your account password has been changed successfully."
            )
        )
        
        return True
    
    async def authenticate_lti_user(
        self, 
        launch_data: LTILaunchRequest, 
        db: AsyncSession
    ) -> User:
        """Authenticate or create user from LTI launch data."""
        # Check if user exists by LTI user ID
        result = await db.execute(
            select(User).where(User.lms_user_id == launch_data.user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            # Update last login
            await db.execute(
                update(User)
                .where(User.id == user.id)
                .values(last_login=datetime.utcnow())
            )
            await db.commit()
            return user
        
        # Create new user from LTI data
        # Extract email from custom parameters or use user_id as fallback
        email = getattr(launch_data, 'lis_person_contact_email_primary', None)
        if not email:
            # Generate email from user_id and tool consumer
            email = f"{launch_data.user_id}@{launch_data.tool_consumer_instance_guid}"
        
        # Extract name
        name = getattr(launch_data, 'lis_person_name_full', launch_data.user_id)
        
        # Determine role
        roles = launch_data.roles.lower() if launch_data.roles else ""
        if any(role in roles for role in ['instructor', 'teacher', 'admin']):
            user_role = UserRole.ADMIN
        else:
            user_role = UserRole.STUDENT
        
        # Create user
        new_user = User(
            email=email,
            full_name=name,
            hashed_password="",  # No password for LTI users
            role=user_role,
            is_active=True,
            is_verified=True,  # LTI users are pre-verified
            lms_user_id=launch_data.user_id,
            lms_context_id=launch_data.context_id,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Create writing profile for students
        if user_role == UserRole.STUDENT:
            writing_profile = WritingProfile(
                student_id=new_user.id,
                confidence_score=0.0,
                sample_count=0,
                created_at=datetime.utcnow()
            )
            db.add(writing_profile)
            await db.commit()
        
        # Log LTI user creation
        await audit_service.log_event(
            db=db,
            event_type=audit_service.EventType.USER_REGISTRATION,
            event_category=audit_service.Category.AUTH,
            description=f"LTI user created: {new_user.id}",
            user_id=new_user.id,
            metadata={
                "lms_user_id": launch_data.user_id,
                "context_id": launch_data.context_id,
                "tool_consumer": launch_data.tool_consumer_instance_guid
            },
            risk_level=audit_service.RiskLevel.LOW
        )
        
        return new_user


# Create global instance
auth_service = AuthService()
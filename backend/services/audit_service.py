"""
Audit logging service for security and compliance monitoring.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text
from sqlalchemy.sql import func
import json

from core.database import Base
from core.redis import cache

logger = logging.getLogger(__name__)


class AuditLog(Base):
    """Audit log model for tracking security events."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)  # Nullable for anonymous events
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)  # AUTH, ACCESS, DATA, SYSTEM
    description = Column(Text, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    event_metadata = Column(JSON, nullable=True)  # Additional event data
    risk_level = Column(String(20), default='LOW')  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type='{self.event_type}', user_id={self.user_id})>"


class AuditService:
    """Service for audit logging and security monitoring."""
    
    # Event types
    class EventType:
        # Authentication events
        LOGIN_SUCCESS = "LOGIN_SUCCESS"
        LOGIN_FAILED = "LOGIN_FAILED"
        LOGOUT = "LOGOUT"
        TOKEN_REFRESH = "TOKEN_REFRESH"
        PASSWORD_CHANGE = "PASSWORD_CHANGE"
        PASSWORD_RESET_REQUEST = "PASSWORD_RESET_REQUEST"
        PASSWORD_RESET_SUCCESS = "PASSWORD_RESET_SUCCESS"
        EMAIL_VERIFICATION = "EMAIL_VERIFICATION"
        ACCOUNT_LOCKED = "ACCOUNT_LOCKED"
        ACCOUNT_UNLOCKED = "ACCOUNT_UNLOCKED"
        
        # Access events
        UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
        PERMISSION_DENIED = "PERMISSION_DENIED"
        ROLE_CHANGE = "ROLE_CHANGE"
        
        # Data events
        SUBMISSION_UPLOAD = "SUBMISSION_UPLOAD"
        SUBMISSION_DELETE = "SUBMISSION_DELETE"
        PROFILE_UPDATE = "PROFILE_UPDATE"
        DATA_EXPORT = "DATA_EXPORT"
        
        # System events
        SYSTEM_ERROR = "SYSTEM_ERROR"
        SECURITY_VIOLATION = "SECURITY_VIOLATION"
        RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Event categories
    class Category:
        AUTH = "AUTH"
        ACCESS = "ACCESS"
        DATA = "DATA"
        SYSTEM = "SYSTEM"
    
    # Risk levels
    class RiskLevel:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"
    
    @staticmethod
    async def log_event(
        db: AsyncSession,
        event_type: str,
        event_category: str,
        description: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        risk_level: str = RiskLevel.LOW
    ) -> None:
        """Log an audit event."""
        try:
            audit_log = AuditLog(
                user_id=user_id,
                event_type=event_type,
                event_category=event_category,
                description=description,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                session_id=session_id,
                event_metadata=metadata,
                risk_level=risk_level
            )
            
            db.add(audit_log)
            await db.commit()
            
            # Cache high-risk events for real-time monitoring
            if risk_level in [AuditService.RiskLevel.HIGH, AuditService.RiskLevel.CRITICAL]:
                await cache.set(
                    f"high_risk_event:{audit_log.id}",
                    {
                        "event_type": event_type,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "risk_level": risk_level,
                        "description": description
                    },
                    expire=3600  # 1 hour
                )
            
            logger.info(f"Audit event logged: {event_type} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    @staticmethod
    async def log_login_success(
        db: AsyncSession,
        user_id: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Log successful login."""
        await AuditService.log_event(
            db=db,
            event_type=AuditService.EventType.LOGIN_SUCCESS,
            event_category=AuditService.Category.AUTH,
            description=f"User {user_id} logged in successfully",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            risk_level=AuditService.RiskLevel.LOW
        )
    
    @staticmethod
    async def log_login_failed(
        db: AsyncSession,
        email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        reason: str = "Invalid credentials"
    ) -> None:
        """Log failed login attempt."""
        await AuditService.log_event(
            db=db,
            event_type=AuditService.EventType.LOGIN_FAILED,
            event_category=AuditService.Category.AUTH,
            description=f"Failed login attempt for email {email}: {reason}",
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"email": email, "reason": reason},
            risk_level=AuditService.RiskLevel.MEDIUM
        )
    
    @staticmethod
    async def log_password_change(
        db: AsyncSession,
        user_id: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log password change."""
        await AuditService.log_event(
            db=db,
            event_type=AuditService.EventType.PASSWORD_CHANGE,
            event_category=AuditService.Category.AUTH,
            description=f"User {user_id} changed password",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_level=AuditService.RiskLevel.MEDIUM
        )
    
    @staticmethod
    async def log_unauthorized_access(
        db: AsyncSession,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> None:
        """Log unauthorized access attempt."""
        await AuditService.log_event(
            db=db,
            event_type=AuditService.EventType.UNAUTHORIZED_ACCESS,
            event_category=AuditService.Category.ACCESS,
            description=f"Unauthorized access attempt to {endpoint or 'unknown endpoint'}",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"endpoint": endpoint},
            risk_level=AuditService.RiskLevel.HIGH
        )
    
    @staticmethod
    async def log_account_locked(
        db: AsyncSession,
        user_id: int,
        reason: str = "Multiple failed login attempts",
        ip_address: Optional[str] = None
    ) -> None:
        """Log account lockout."""
        await AuditService.log_event(
            db=db,
            event_type=AuditService.EventType.ACCOUNT_LOCKED,
            event_category=AuditService.Category.AUTH,
            description=f"Account {user_id} locked: {reason}",
            user_id=user_id,
            ip_address=ip_address,
            metadata={"reason": reason},
            risk_level=AuditService.RiskLevel.HIGH
        )
    
    @staticmethod
    async def check_suspicious_activity(
        db: AsyncSession,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        time_window_minutes: int = 15
    ) -> Dict[str, Any]:
        """Check for suspicious activity patterns."""
        # This would typically involve complex queries to detect patterns
        # For now, we'll implement basic checks using Redis counters
        
        suspicious_indicators = {
            "multiple_failed_logins": False,
            "rapid_requests": False,
            "unusual_ip": False,
            "risk_score": 0
        }
        
        try:
            if user_id:
                # Check failed login attempts
                failed_login_key = f"failed_logins:{user_id}"
                failed_count = await cache.get(failed_login_key) or 0
                if isinstance(failed_count, str):
                    failed_count = int(failed_count)
                
                if failed_count >= 5:
                    suspicious_indicators["multiple_failed_logins"] = True
                    suspicious_indicators["risk_score"] += 30
            
            if ip_address:
                # Check request rate from IP
                request_key = f"requests:{ip_address}"
                request_count = await cache.get(request_key) or 0
                if isinstance(request_count, str):
                    request_count = int(request_count)
                
                if request_count > 100:  # More than 100 requests in time window
                    suspicious_indicators["rapid_requests"] = True
                    suspicious_indicators["risk_score"] += 20
            
            return suspicious_indicators
            
        except Exception as e:
            logger.error(f"Error checking suspicious activity: {e}")
            return suspicious_indicators


# Global audit service instance
audit_service = AuditService()
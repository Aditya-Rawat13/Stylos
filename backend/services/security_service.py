"""
Security service for account protection and threat detection.
"""
import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from fastapi import HTTPException, status
import asyncio
import json

from core.database import Base
from core.redis import cache
from models.user import User
from services.audit_service import audit_service
from services.email_service import email_service
from utils.encryption import DataAnonymizer

logger = logging.getLogger(__name__)


class SecurityIncident(Base):
    """Model for tracking security incidents."""
    __tablename__ = "security_incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    status = Column(String(20), default="OPEN", nullable=False)  # OPEN, INVESTIGATING, RESOLVED, CLOSED
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    affected_user_id = Column(Integer, nullable=True)
    source_ip = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    detection_method = Column(String(100), nullable=False)
    incident_metadata = Column(Text, nullable=True)  # JSON metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(Integer, nullable=True)
    resolution_notes = Column(Text, nullable=True)


class ThreatIntelligence(Base):
    """Model for storing threat intelligence data."""
    __tablename__ = "threat_intelligence"
    
    id = Column(Integer, primary_key=True, index=True)
    indicator_type = Column(String(50), nullable=False, index=True)  # IP, EMAIL, USER_AGENT, etc.
    indicator_value = Column(String(500), nullable=False, index=True)
    threat_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    source = Column(String(100), nullable=False)
    confidence = Column(Integer, nullable=False)  # 0-100
    is_active = Column(Boolean, default=True, nullable=False)
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    threat_metadata = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)


class SecurityService:
    """Service for account security and threat detection."""
    
    # Security thresholds
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW_MINUTES = 15
    
    # Intrusion detection thresholds
    SUSPICIOUS_LOGIN_THRESHOLD = 10  # Failed logins from same IP
    RAPID_REQUEST_THRESHOLD = 50     # Requests per minute
    GEOGRAPHIC_ANOMALY_THRESHOLD = 1000  # km distance
    
    # Threat intelligence
    KNOWN_MALICIOUS_IPS = set()
    SUSPICIOUS_USER_AGENTS = [
        r'.*bot.*', r'.*crawler.*', r'.*spider.*', r'.*scraper.*',
        r'.*scanner.*', r'.*exploit.*', r'.*hack.*'
    ]
    
    @staticmethod
    async def track_login_attempt(
        db: AsyncSession,
        email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = False
    ) -> Dict[str, Any]:
        """Track login attempt and check for account lockout."""
        result = {
            "locked": False,
            "attempts_remaining": SecurityService.MAX_LOGIN_ATTEMPTS,
            "lockout_expires": None,
            "should_alert": False
        }
        
        try:
            # Get user to check if account exists
            user_result = await db.execute(select(User).where(User.email == email))
            user = user_result.scalar_one_or_none()
            
            if success:
                # Clear failed attempts on successful login
                if user:
                    await cache.delete(f"failed_logins:{user.id}")
                    await cache.delete(f"lockout:{user.id}")
                    await audit_service.log_login_success(
                        db, user.id, ip_address, user_agent
                    )
                return result
            
            # Handle failed login
            if user:
                # Increment failed login counter
                failed_key = f"failed_logins:{user.id}"
                failed_count = await cache.get(failed_key) or 0
                if isinstance(failed_count, str):
                    failed_count = int(failed_count)
                
                failed_count += 1
                await cache.set(
                    failed_key, 
                    failed_count, 
                    expire=timedelta(minutes=SecurityService.LOCKOUT_DURATION_MINUTES)
                )
                
                result["attempts_remaining"] = max(0, SecurityService.MAX_LOGIN_ATTEMPTS - failed_count)
                
                # Check if account should be locked
                if failed_count >= SecurityService.MAX_LOGIN_ATTEMPTS:
                    lockout_expires = datetime.utcnow() + timedelta(
                        minutes=SecurityService.LOCKOUT_DURATION_MINUTES
                    )
                    
                    await cache.set(
                        f"lockout:{user.id}",
                        lockout_expires.isoformat(),
                        expire=timedelta(minutes=SecurityService.LOCKOUT_DURATION_MINUTES)
                    )
                    
                    result["locked"] = True
                    result["lockout_expires"] = lockout_expires
                    result["should_alert"] = True
                    
                    # Log account lockout
                    await audit_service.log_account_locked(
                        db, user.id, "Multiple failed login attempts", ip_address
                    )
                    
                    # Send security alert email
                    asyncio.create_task(
                        email_service.send_security_alert(
                            user.email,
                            user.full_name,
                            "Account Locked",
                            f"Your account has been locked due to {failed_count} failed login attempts. "
                            f"It will be unlocked automatically at {lockout_expires.strftime('%Y-%m-%d %H:%M:%S UTC')}."
                        )
                    )
            
            # Log failed login attempt
            await audit_service.log_login_failed(
                db, email, ip_address, user_agent
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error tracking login attempt: {e}")
            return result
    
    @staticmethod
    async def is_account_locked(user_id: int) -> Tuple[bool, Optional[datetime]]:
        """Check if account is currently locked."""
        try:
            lockout_data = await cache.get(f"lockout:{user_id}")
            if not lockout_data:
                return False, None
            
            if isinstance(lockout_data, str):
                lockout_expires = datetime.fromisoformat(lockout_data)
            else:
                return False, None
            
            if datetime.utcnow() > lockout_expires:
                # Lockout expired, clean up
                await cache.delete(f"lockout:{user_id}")
                await cache.delete(f"failed_logins:{user_id}")
                return False, None
            
            return True, lockout_expires
            
        except Exception as e:
            logger.error(f"Error checking account lock status: {e}")
            return False, None
    
    @staticmethod
    async def unlock_account(db: AsyncSession, user_id: int, admin_user_id: Optional[int] = None) -> bool:
        """Manually unlock an account."""
        try:
            # Remove lockout and failed login counters
            await cache.delete(f"lockout:{user_id}")
            await cache.delete(f"failed_logins:{user_id}")
            
            # Log unlock event
            await audit_service.log_event(
                db=db,
                event_type=audit_service.EventType.ACCOUNT_UNLOCKED,
                event_category=audit_service.Category.AUTH,
                description=f"Account {user_id} unlocked by admin {admin_user_id or 'system'}",
                user_id=user_id,
                incident_metadata=json.dumps({"unlocked_by": admin_user_id}),
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error unlocking account {user_id}: {e}")
            return False
    
    @staticmethod
    async def check_rate_limit(
        identifier: str,  # IP address or user ID
        limit_type: str = "general"
    ) -> Tuple[bool, int, int]:
        """
        Check rate limiting for requests.
        Returns: (is_limited, current_count, limit)
        """
        try:
            key = f"rate_limit:{limit_type}:{identifier}"
            current_count = await cache.get(key) or 0
            if isinstance(current_count, str):
                current_count = int(current_count)
            
            limit = SecurityService.RATE_LIMIT_REQUESTS
            
            if current_count >= limit:
                return True, current_count, limit
            
            # Increment counter
            await cache.set(
                key,
                current_count + 1,
                expire=timedelta(minutes=SecurityService.RATE_LIMIT_WINDOW_MINUTES)
            )
            
            return False, current_count + 1, limit
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False, 0, SecurityService.RATE_LIMIT_REQUESTS
    
    @staticmethod
    async def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength and return detailed feedback."""
        result = {
            "is_valid": True,
            "score": 0,
            "feedback": [],
            "requirements_met": {
                "min_length": False,
                "has_uppercase": False,
                "has_lowercase": False,
                "has_digit": False,
                "has_special": False,
                "no_common_patterns": True
            }
        }
        
        # Check minimum length
        if len(password) >= 8:
            result["requirements_met"]["min_length"] = True
            result["score"] += 20
        else:
            result["is_valid"] = False
            result["feedback"].append("Password must be at least 8 characters long")
        
        # Check for uppercase letter
        if any(c.isupper() for c in password):
            result["requirements_met"]["has_uppercase"] = True
            result["score"] += 15
        else:
            result["is_valid"] = False
            result["feedback"].append("Password must contain at least one uppercase letter")
        
        # Check for lowercase letter
        if any(c.islower() for c in password):
            result["requirements_met"]["has_lowercase"] = True
            result["score"] += 15
        else:
            result["is_valid"] = False
            result["feedback"].append("Password must contain at least one lowercase letter")
        
        # Check for digit
        if any(c.isdigit() for c in password):
            result["requirements_met"]["has_digit"] = True
            result["score"] += 15
        else:
            result["is_valid"] = False
            result["feedback"].append("Password must contain at least one digit")
        
        # Check for special character
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if any(c in special_chars for c in password):
            result["requirements_met"]["has_special"] = True
            result["score"] += 20
        else:
            result["feedback"].append("Consider adding special characters for stronger security")
        
        # Check for common patterns
        common_patterns = [
            "password", "123456", "qwerty", "admin", "user",
            "login", "welcome", "letmein", "monkey", "dragon"
        ]
        
        password_lower = password.lower()
        for pattern in common_patterns:
            if pattern in password_lower:
                result["requirements_met"]["no_common_patterns"] = False
                result["score"] = max(0, result["score"] - 30)
                result["feedback"].append("Avoid common words and patterns")
                break
        
        # Additional length bonus
        if len(password) >= 12:
            result["score"] += 10
        if len(password) >= 16:
            result["score"] += 5
        
        # Cap score at 100
        result["score"] = min(100, result["score"])
        
        return result
    
    @staticmethod
    async def detect_suspicious_patterns(
        db: AsyncSession,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect suspicious activity patterns."""
        return await audit_service.check_suspicious_activity(
            db, user_id, ip_address
        )
    
    @staticmethod
    async def require_account_verification(user: User) -> None:
        """Check if user account is verified and active."""
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email verification required"
            )
        
        # Check if account is locked
        is_locked, lockout_expires = await SecurityService.is_account_locked(user.id)
        if is_locked:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account is locked until {lockout_expires.strftime('%Y-%m-%d %H:%M:%S UTC') if lockout_expires else 'unknown'}"
            )
    
    @staticmethod
    async def detect_intrusion_attempt(
        db: AsyncSession,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[int] = None,
        request_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect potential intrusion attempts."""
        threats_detected = []
        risk_score = 0
        
        try:
            # Check against threat intelligence
            if ip_address:
                ip_threat = await SecurityService._check_ip_reputation(db, ip_address)
                if ip_threat:
                    threats_detected.append(ip_threat)
                    risk_score += ip_threat.get('risk_score', 0)
            
            # Check user agent patterns
            if user_agent:
                ua_threat = SecurityService._check_user_agent_patterns(user_agent)
                if ua_threat:
                    threats_detected.append(ua_threat)
                    risk_score += ua_threat.get('risk_score', 0)
            
            # Check for rapid requests from same IP
            if ip_address:
                rapid_requests = await SecurityService._check_rapid_requests(ip_address)
                if rapid_requests:
                    threats_detected.append(rapid_requests)
                    risk_score += rapid_requests.get('risk_score', 0)
            
            # Check for suspicious login patterns
            if user_id:
                login_anomaly = await SecurityService._check_login_anomalies(db, user_id, ip_address)
                if login_anomaly:
                    threats_detected.append(login_anomaly)
                    risk_score += login_anomaly.get('risk_score', 0)
            
            # Create security incident if high risk
            if risk_score >= 70:
                await SecurityService._create_security_incident(
                    db, threats_detected, ip_address, user_agent, user_id
                )
            
            return {
                "threats_detected": threats_detected,
                "risk_score": min(risk_score, 100),
                "action_required": risk_score >= 70
            }
            
        except Exception as e:
            logger.error(f"Error in intrusion detection: {e}")
            return {"threats_detected": [], "risk_score": 0, "action_required": False}
    
    @staticmethod
    async def _check_ip_reputation(db: AsyncSession, ip_address: str) -> Optional[Dict]:
        """Check IP address against threat intelligence."""
        try:
            # Check cached reputation
            cached_rep = await cache.get(f"ip_reputation:{ip_address}")
            if cached_rep:
                return json.loads(cached_rep) if isinstance(cached_rep, str) else cached_rep
            
            # Check database threat intelligence
            result = await db.execute(
                select(ThreatIntelligence)
                .where(
                    ThreatIntelligence.indicator_type == "IP",
                    ThreatIntelligence.indicator_value == ip_address,
                    ThreatIntelligence.is_active == True
                )
            )
            
            threat_record = result.scalar_one_or_none()
            if threat_record:
                threat_data = {
                    "type": "malicious_ip",
                    "description": f"Known malicious IP: {threat_record.threat_type}",
                    "risk_score": min(threat_record.confidence, 50),
                    "source": threat_record.source
                }
                
                # Cache for 1 hour
                await cache.set(f"ip_reputation:{ip_address}", json.dumps(threat_data), expire=3600)
                return threat_data
            
            # Check for geographic anomalies (simplified)
            geo_anomaly = await SecurityService._check_geographic_anomaly(ip_address)
            if geo_anomaly:
                return geo_anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking IP reputation: {e}")
            return None
    
    @staticmethod
    def _check_user_agent_patterns(user_agent: str) -> Optional[Dict]:
        """Check user agent for suspicious patterns."""
        try:
            user_agent_lower = user_agent.lower()
            
            for pattern in SecurityService.SUSPICIOUS_USER_AGENTS:
                if re.match(pattern, user_agent_lower):
                    return {
                        "type": "suspicious_user_agent",
                        "description": f"Suspicious user agent pattern detected",
                        "risk_score": 30,
                        "pattern": pattern
                    }
            
            # Check for empty or very short user agents
            if len(user_agent.strip()) < 10:
                return {
                    "type": "suspicious_user_agent",
                    "description": "Unusually short user agent",
                    "risk_score": 20
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking user agent patterns: {e}")
            return None
    
    @staticmethod
    async def _check_rapid_requests(ip_address: str) -> Optional[Dict]:
        """Check for rapid requests from IP address."""
        try:
            # Get request count for last minute
            key = f"request_count_minute:{ip_address}"
            count = await cache.get(key) or 0
            if isinstance(count, str):
                count = int(count)
            
            if count > SecurityService.RAPID_REQUEST_THRESHOLD:
                return {
                    "type": "rapid_requests",
                    "description": f"Rapid requests detected: {count} requests/minute",
                    "risk_score": min(40 + (count - SecurityService.RAPID_REQUEST_THRESHOLD), 60),
                    "request_count": count
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking rapid requests: {e}")
            return None
    
    @staticmethod
    async def _check_login_anomalies(
        db: AsyncSession, 
        user_id: int, 
        ip_address: Optional[str]
    ) -> Optional[Dict]:
        """Check for login anomalies."""
        try:
            # Check for multiple failed logins
            failed_count = await cache.get(f"failed_logins:{user_id}") or 0
            if isinstance(failed_count, str):
                failed_count = int(failed_count)
            
            if failed_count >= 3:
                return {
                    "type": "multiple_failed_logins",
                    "description": f"Multiple failed login attempts: {failed_count}",
                    "risk_score": min(20 + (failed_count * 5), 40),
                    "failed_count": failed_count
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking login anomalies: {e}")
            return None
    
    @staticmethod
    async def _check_geographic_anomaly(ip_address: str) -> Optional[Dict]:
        """Check for geographic anomalies (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In production, use a proper GeoIP service
            
            # Check if IP is from known high-risk countries/regions
            # This would typically involve a GeoIP lookup
            
            # For now, just check for common VPN/proxy patterns
            if any(pattern in ip_address for pattern in ['10.', '192.168.', '172.']):
                return {
                    "type": "private_ip_access",
                    "description": "Access from private IP range",
                    "risk_score": 15
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking geographic anomaly: {e}")
            return None
    
    @staticmethod
    async def _create_security_incident(
        db: AsyncSession,
        threats: List[Dict],
        ip_address: Optional[str],
        user_agent: Optional[str],
        user_id: Optional[int]
    ) -> None:
        """Create security incident record."""
        try:
            # Determine severity based on risk score
            total_risk = sum(threat.get('risk_score', 0) for threat in threats)
            if total_risk >= 90:
                severity = "CRITICAL"
            elif total_risk >= 70:
                severity = "HIGH"
            elif total_risk >= 50:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            # Create incident
            incident = SecurityIncident(
                incident_type="INTRUSION_ATTEMPT",
                severity=severity,
                title=f"Intrusion attempt detected from {DataAnonymizer.anonymize_ip_address(ip_address or 'unknown')}",
                description=f"Multiple security threats detected: {', '.join([t.get('type', 'unknown') for t in threats])}",
                affected_user_id=user_id,
                source_ip=ip_address,
                user_agent=user_agent,
                detection_method="automated_analysis",
                incident_metadata=json.dumps({
                    "threats": threats,
                    "total_risk_score": total_risk,
                    "detection_timestamp": datetime.utcnow().isoformat()
                })
            )
            
            db.add(incident)
            await db.commit()
            
            # Log incident
            await audit_service.log_event(
                db=db,
                event_type="SECURITY_INCIDENT_CREATED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Security incident {incident.id} created: {incident.title}",
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "incident_id": incident.id,
                    "severity": severity,
                    "threats_count": len(threats)
                },
                risk_level=audit_service.RiskLevel.HIGH
            )
            
            logger.warning(f"Security incident created: {incident.id} - {incident.title}")
            
        except Exception as e:
            logger.error(f"Error creating security incident: {e}")
    
    @staticmethod
    async def add_threat_intelligence(
        db: AsyncSession,
        indicator_type: str,
        indicator_value: str,
        threat_type: str,
        severity: str,
        source: str = "manual",
        confidence: int = 80,
        expires_days: Optional[int] = None
    ) -> bool:
        """Add threat intelligence indicator."""
        try:
            expires_at = None
            if expires_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
            threat_intel = ThreatIntelligence(
                indicator_type=indicator_type,
                indicator_value=indicator_value,
                threat_type=threat_type,
                severity=severity,
                source=source,
                confidence=confidence,
                expires_at=expires_at
            )
            
            db.add(threat_intel)
            await db.commit()
            
            # Clear related caches
            if indicator_type == "IP":
                await cache.delete(f"ip_reputation:{indicator_value}")
            
            # Log addition
            await audit_service.log_event(
                db=db,
                event_type="THREAT_INTEL_ADDED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Threat intelligence added: {indicator_type} - {indicator_value}",
                metadata={
                    "indicator_type": indicator_type,
                    "threat_type": threat_type,
                    "severity": severity,
                    "source": source
                },
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding threat intelligence: {e}")
            return False
    
    @staticmethod
    async def get_security_dashboard_data(db: AsyncSession) -> Dict[str, Any]:
        """Get security dashboard data."""
        try:
            # Get recent incidents
            recent_incidents = await db.execute(
                select(SecurityIncident)
                .where(SecurityIncident.created_at >= datetime.utcnow() - timedelta(days=7))
                .order_by(SecurityIncident.created_at.desc())
                .limit(10)
            )
            
            incidents = []
            for incident in recent_incidents.scalars():
                incidents.append({
                    "id": incident.id,
                    "type": incident.incident_type,
                    "severity": incident.severity,
                    "title": incident.title,
                    "created_at": incident.created_at.isoformat(),
                    "status": incident.status
                })
            
            # Get threat intelligence stats
            threat_intel_count = await db.execute(
                select(func.count(ThreatIntelligence.id))
                .where(ThreatIntelligence.is_active == True)
            )
            
            # Get failed login attempts (last 24h)
            failed_logins_key = "failed_logins_24h"
            failed_logins = await cache.get(failed_logins_key) or 0
            
            return {
                "recent_incidents": incidents,
                "active_threat_indicators": threat_intel_count.scalar(),
                "failed_logins_24h": failed_logins,
                "security_status": "NORMAL"  # Could be calculated based on recent activity
            }
            
        except Exception as e:
            logger.error(f"Error getting security dashboard data: {e}")
            return {}


# Global security service instance
security_service = SecurityService()
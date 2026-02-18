"""
Key management service for secure key storage and rotation.
"""
import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.sql import func
import logging

from core.database import Base
from core.redis import cache
from utils.encryption import EncryptionService
from services.audit_service import audit_service

logger = logging.getLogger(__name__)


class EncryptionKey(Base):
    """Model for storing encryption key metadata."""
    __tablename__ = "encryption_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(100), unique=True, nullable=False, index=True)
    key_type = Column(String(50), nullable=False)  # master, data, session
    algorithm = Column(String(50), nullable=False, default="AES-256-GCM")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    rotated_at = Column(DateTime(timezone=True), nullable=True)
    rotation_reason = Column(String(200), nullable=True)
    key_metadata = Column(Text, nullable=True)  # JSON metadata


class KeyManagementService:
    """Service for managing encryption keys and rotation."""
    
    # Key rotation intervals
    MASTER_KEY_ROTATION_DAYS = 365  # 1 year
    DATA_KEY_ROTATION_DAYS = 90     # 3 months
    SESSION_KEY_ROTATION_HOURS = 24  # 1 day
    
    def __init__(self):
        self.encryption_service = EncryptionService()
    
    async def generate_master_key(self, db: AsyncSession) -> str:
        """Generate new master encryption key."""
        try:
            # Generate cryptographically secure key
            key_bytes = secrets.token_bytes(32)  # 256-bit key
            key_b64 = secrets.token_urlsafe(32)
            
            # Create key record
            key_id = f"master_{secrets.token_hex(8)}"
            expires_at = datetime.utcnow() + timedelta(days=self.MASTER_KEY_ROTATION_DAYS)
            
            key_record = EncryptionKey(
                key_id=key_id,
                key_type="master",
                algorithm="AES-256-GCM",
                is_active=True,
                expires_at=expires_at,
                key_metadata=json.dumps({
                    "purpose": "master_encryption",
                    "key_length": 256,
                    "generated_by": "system"
                })
            )
            
            db.add(key_record)
            await db.commit()
            
            # Store key securely (in production, use HSM or key vault)
            await cache.set(f"master_key:{key_id}", key_b64, expire=86400 * 365)
            
            # Log key generation
            await audit_service.log_event(
                db=db,
                event_type="KEY_GENERATED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Master key {key_id} generated",
                metadata={"key_id": key_id, "key_type": "master"},
                risk_level=audit_service.RiskLevel.HIGH
            )
            
            logger.info(f"Master key {key_id} generated successfully")
            return key_id
            
        except Exception as e:
            logger.error(f"Failed to generate master key: {e}")
            raise
    
    async def rotate_master_key(
        self, 
        db: AsyncSession, 
        current_key_id: str,
        reason: str = "scheduled_rotation"
    ) -> str:
        """Rotate master encryption key."""
        try:
            # Generate new master key
            new_key_id = await self.generate_master_key(db)
            
            # Deactivate old key
            await self.deactivate_key(db, current_key_id, reason)
            
            # Log rotation
            await audit_service.log_event(
                db=db,
                event_type="KEY_ROTATED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Master key rotated from {current_key_id} to {new_key_id}",
                metadata={
                    "old_key_id": current_key_id,
                    "new_key_id": new_key_id,
                    "reason": reason
                },
                risk_level=audit_service.RiskLevel.HIGH
            )
            
            logger.info(f"Master key rotated: {current_key_id} -> {new_key_id}")
            return new_key_id
            
        except Exception as e:
            logger.error(f"Failed to rotate master key: {e}")
            raise
    
    async def generate_data_encryption_key(self, db: AsyncSession, purpose: str) -> str:
        """Generate data encryption key for specific purpose."""
        try:
            # Generate DEK
            key_bytes = secrets.token_bytes(32)
            key_b64 = secrets.token_urlsafe(32)
            
            key_id = f"dek_{purpose}_{secrets.token_hex(6)}"
            expires_at = datetime.utcnow() + timedelta(days=self.DATA_KEY_ROTATION_DAYS)
            
            key_record = EncryptionKey(
                key_id=key_id,
                key_type="data",
                algorithm="AES-256-GCM",
                is_active=True,
                expires_at=expires_at,
                key_metadata=json.dumps({
                    "purpose": purpose,
                    "key_length": 256,
                    "generated_by": "system"
                })
            )
            
            db.add(key_record)
            await db.commit()
            
            # Store encrypted DEK (encrypt with master key)
            encrypted_key = self.encryption_service.encrypt_text(key_b64)
            await cache.set(f"data_key:{key_id}", encrypted_key, expire=86400 * 90)
            
            # Log key generation
            await audit_service.log_event(
                db=db,
                event_type="DATA_KEY_GENERATED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Data encryption key {key_id} generated for {purpose}",
                metadata={"key_id": key_id, "purpose": purpose},
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return key_id
            
        except Exception as e:
            logger.error(f"Failed to generate data encryption key: {e}")
            raise
    
    async def get_active_key(self, db: AsyncSession, key_type: str) -> Optional[str]:
        """Get active key ID for specified type."""
        try:
            from sqlalchemy import select
            
            result = await db.execute(
                select(EncryptionKey)
                .where(
                    EncryptionKey.key_type == key_type,
                    EncryptionKey.is_active == True,
                    EncryptionKey.expires_at > datetime.utcnow()
                )
                .order_by(EncryptionKey.created_at.desc())
                .limit(1)
            )
            
            key_record = result.scalar_one_or_none()
            return key_record.key_id if key_record else None
            
        except Exception as e:
            logger.error(f"Failed to get active key: {e}")
            return None
    
    async def deactivate_key(
        self, 
        db: AsyncSession, 
        key_id: str, 
        reason: str = "rotation"
    ) -> bool:
        """Deactivate encryption key."""
        try:
            from sqlalchemy import update
            
            # Update key record
            await db.execute(
                update(EncryptionKey)
                .where(EncryptionKey.key_id == key_id)
                .values(
                    is_active=False,
                    rotated_at=datetime.utcnow(),
                    rotation_reason=reason
                )
            )
            await db.commit()
            
            # Remove from cache (keep for grace period)
            await cache.expire(f"master_key:{key_id}", 86400 * 7)  # 7 days grace
            await cache.expire(f"data_key:{key_id}", 86400 * 7)
            
            # Log deactivation
            await audit_service.log_event(
                db=db,
                event_type="KEY_DEACTIVATED",
                event_category=audit_service.Category.SYSTEM,
                description=f"Key {key_id} deactivated: {reason}",
                metadata={"key_id": key_id, "reason": reason},
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate key {key_id}: {e}")
            return False
    
    async def check_key_expiration(self, db: AsyncSession) -> List[Dict]:
        """Check for keys approaching expiration."""
        try:
            from sqlalchemy import select
            
            # Check for keys expiring in next 30 days
            expiry_threshold = datetime.utcnow() + timedelta(days=30)
            
            result = await db.execute(
                select(EncryptionKey)
                .where(
                    EncryptionKey.is_active == True,
                    EncryptionKey.expires_at <= expiry_threshold
                )
            )
            
            expiring_keys = []
            for key_record in result.scalars():
                days_until_expiry = (key_record.expires_at - datetime.utcnow()).days
                expiring_keys.append({
                    "key_id": key_record.key_id,
                    "key_type": key_record.key_type,
                    "expires_at": key_record.expires_at,
                    "days_until_expiry": days_until_expiry
                })
            
            return expiring_keys
            
        except Exception as e:
            logger.error(f"Failed to check key expiration: {e}")
            return []
    
    async def emergency_key_revocation(
        self, 
        db: AsyncSession, 
        key_id: str, 
        reason: str = "security_incident"
    ) -> bool:
        """Emergency revocation of compromised key."""
        try:
            # Immediately deactivate key
            success = await self.deactivate_key(db, key_id, f"EMERGENCY: {reason}")
            
            if success:
                # Remove from cache immediately
                await cache.delete(f"master_key:{key_id}")
                await cache.delete(f"data_key:{key_id}")
                
                # Log emergency revocation
                await audit_service.log_event(
                    db=db,
                    event_type="KEY_EMERGENCY_REVOKED",
                    event_category=audit_service.Category.SYSTEM,
                    description=f"Emergency revocation of key {key_id}: {reason}",
                    metadata={"key_id": key_id, "reason": reason},
                    risk_level=audit_service.RiskLevel.CRITICAL
                )
                
                logger.critical(f"Emergency key revocation: {key_id} - {reason}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed emergency key revocation: {e}")
            return False
    
    async def get_key_statistics(self, db: AsyncSession) -> Dict:
        """Get key management statistics."""
        try:
            from sqlalchemy import select, func as sql_func
            
            # Count active keys by type
            result = await db.execute(
                select(
                    EncryptionKey.key_type,
                    sql_func.count(EncryptionKey.id).label('count')
                )
                .where(EncryptionKey.is_active == True)
                .group_by(EncryptionKey.key_type)
            )
            
            active_keys = {row.key_type: row.count for row in result}
            
            # Count total keys
            total_result = await db.execute(
                select(sql_func.count(EncryptionKey.id))
            )
            total_keys = total_result.scalar()
            
            # Get expiring keys
            expiring_keys = await self.check_key_expiration(db)
            
            return {
                "active_keys": active_keys,
                "total_keys": total_keys,
                "expiring_keys_count": len(expiring_keys),
                "expiring_keys": expiring_keys
            }
            
        except Exception as e:
            logger.error(f"Failed to get key statistics: {e}")
            return {}


# Global key management service
key_management_service = KeyManagementService()
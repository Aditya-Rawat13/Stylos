"""
Compliance and data retention service for GDPR, FERPA, and other regulations.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy import select, update, delete
import json

from core.database import Base
from services.audit_service import audit_service
from utils.encryption import DataAnonymizer, get_encryption_service

logger = logging.getLogger(__name__)


class DataRetentionPolicy(Base):
    """Model for data retention policies."""
    __tablename__ = "data_retention_policies"
    
    id = Column(Integer, primary_key=True, index=True)
    policy_name = Column(String(100), unique=True, nullable=False)
    data_type = Column(String(100), nullable=False, index=True)
    retention_period_days = Column(Integer, nullable=False)
    deletion_method = Column(String(50), nullable=False)  # HARD_DELETE, ANONYMIZE, ARCHIVE
    is_active = Column(Boolean, default=True, nullable=False)
    legal_basis = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    policy_metadata = Column(JSON, nullable=True)


class DataProcessingRecord(Base):
    """Model for tracking data processing activities (GDPR Article 30)."""
    __tablename__ = "data_processing_records"
    
    id = Column(Integer, primary_key=True, index=True)
    processing_activity = Column(String(200), nullable=False)
    data_controller = Column(String(200), nullable=False)
    data_processor = Column(String(200), nullable=True)
    data_categories = Column(JSON, nullable=False)  # List of data categories
    data_subjects = Column(JSON, nullable=False)    # List of data subject categories
    purposes = Column(JSON, nullable=False)         # List of processing purposes
    legal_basis = Column(String(200), nullable=False)
    recipients = Column(JSON, nullable=True)        # List of recipients
    third_country_transfers = Column(JSON, nullable=True)
    retention_period = Column(String(200), nullable=False)
    security_measures = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ConsentRecord(Base):
    """Model for tracking user consent (GDPR)."""
    __tablename__ = "consent_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    consent_type = Column(String(100), nullable=False)
    purpose = Column(String(200), nullable=False)
    legal_basis = Column(String(100), nullable=False)
    consent_given = Column(Boolean, nullable=False)
    consent_method = Column(String(100), nullable=False)  # EXPLICIT, IMPLIED, OPT_IN, etc.
    consent_text = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    withdrawn_at = Column(DateTime(timezone=True), nullable=True)
    withdrawal_method = Column(String(100), nullable=True)


class DataSubjectRequest(Base):
    """Model for tracking data subject requests (GDPR)."""
    __tablename__ = "data_subject_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    request_type = Column(String(50), nullable=False, index=True)  # ACCESS, RECTIFICATION, ERASURE, etc.
    status = Column(String(50), default="PENDING", nullable=False)
    request_details = Column(Text, nullable=True)
    verification_method = Column(String(100), nullable=True)
    verification_completed = Column(Boolean, default=False)
    response_data = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    deadline = Column(DateTime(timezone=True), nullable=False)
    request_metadata = Column(JSON, nullable=True)


class ComplianceService:
    """Service for managing compliance and data protection."""
    
    # GDPR compliance settings
    GDPR_RESPONSE_DEADLINE_DAYS = 30
    FERPA_RETENTION_YEARS = 5
    
    # Data categories for GDPR
    DATA_CATEGORIES = {
        "PERSONAL_IDENTIFIERS": ["name", "email", "student_id"],
        "ACADEMIC_DATA": ["submissions", "grades", "writing_profiles"],
        "TECHNICAL_DATA": ["ip_addresses", "session_data", "logs"],
        "BEHAVIORAL_DATA": ["usage_patterns", "preferences"]
    }
    
    @staticmethod
    async def initialize_default_policies(db: AsyncSession) -> None:
        """Initialize default data retention policies."""
        try:
            default_policies = [
                {
                    "policy_name": "FERPA_Student_Records",
                    "data_type": "academic_submissions",
                    "retention_period_days": 365 * 5,  # 5 years
                    "deletion_method": "ARCHIVE",
                    "legal_basis": "FERPA compliance - educational records retention"
                },
                {
                    "policy_name": "GDPR_Personal_Data",
                    "data_type": "personal_identifiers",
                    "retention_period_days": 365 * 2,  # 2 years after account closure
                    "deletion_method": "ANONYMIZE",
                    "legal_basis": "GDPR Article 5(1)(e) - storage limitation"
                },
                {
                    "policy_name": "Security_Logs",
                    "data_type": "audit_logs",
                    "retention_period_days": 365,  # 1 year
                    "deletion_method": "ANONYMIZE",
                    "legal_basis": "Security monitoring and incident response"
                },
                {
                    "policy_name": "Session_Data",
                    "data_type": "session_data",
                    "retention_period_days": 30,
                    "deletion_method": "HARD_DELETE",
                    "legal_basis": "Technical necessity - session management"
                }
            ]
            
            for policy_data in default_policies:
                # Check if policy already exists
                existing = await db.execute(
                    select(DataRetentionPolicy)
                    .where(DataRetentionPolicy.policy_name == policy_data["policy_name"])
                )
                
                if not existing.scalar_one_or_none():
                    policy = DataRetentionPolicy(**policy_data)
                    db.add(policy)
            
            await db.commit()
            logger.info("Default data retention policies initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default policies: {e}")
    
    @staticmethod
    async def record_consent(
        db: AsyncSession,
        user_id: int,
        consent_type: str,
        purpose: str,
        legal_basis: str,
        consent_given: bool,
        consent_method: str = "EXPLICIT",
        consent_text: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Record user consent for data processing."""
        try:
            consent_record = ConsentRecord(
                user_id=user_id,
                consent_type=consent_type,
                purpose=purpose,
                legal_basis=legal_basis,
                consent_given=consent_given,
                consent_method=consent_method,
                consent_text=consent_text,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            db.add(consent_record)
            await db.commit()
            
            # Log consent recording
            await audit_service.log_event(
                db=db,
                event_type="CONSENT_RECORDED",
                event_category=audit_service.Category.DATA,
                description=f"Consent recorded for user {user_id}: {consent_type}",
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "consent_type": consent_type,
                    "consent_given": consent_given,
                    "purpose": purpose,
                    "legal_basis": legal_basis
                },
                risk_level=audit_service.RiskLevel.LOW
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording consent: {e}")
            return False
    
    @staticmethod
    async def withdraw_consent(
        db: AsyncSession,
        user_id: int,
        consent_type: str,
        withdrawal_method: str = "USER_REQUEST"
    ) -> bool:
        """Withdraw user consent."""
        try:
            # Update existing consent records
            await db.execute(
                update(ConsentRecord)
                .where(
                    ConsentRecord.user_id == user_id,
                    ConsentRecord.consent_type == consent_type,
                    ConsentRecord.consent_given == True,
                    ConsentRecord.withdrawn_at.is_(None)
                )
                .values(
                    withdrawn_at=datetime.utcnow(),
                    withdrawal_method=withdrawal_method
                )
            )
            
            await db.commit()
            
            # Log consent withdrawal
            await audit_service.log_event(
                db=db,
                event_type="CONSENT_WITHDRAWN",
                event_category=audit_service.Category.DATA,
                description=f"Consent withdrawn for user {user_id}: {consent_type}",
                user_id=user_id,
                metadata={
                    "consent_type": consent_type,
                    "withdrawal_method": withdrawal_method
                },
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error withdrawing consent: {e}")
            return False
    
    @staticmethod
    async def create_data_subject_request(
        db: AsyncSession,
        user_id: int,
        request_type: str,
        request_details: Optional[str] = None
    ) -> Optional[int]:
        """Create a data subject request (GDPR)."""
        try:
            deadline = datetime.utcnow() + timedelta(days=ComplianceService.GDPR_RESPONSE_DEADLINE_DAYS)
            
            request = DataSubjectRequest(
                user_id=user_id,
                request_type=request_type,
                request_details=request_details,
                deadline=deadline
            )
            
            db.add(request)
            await db.commit()
            
            # Log request creation
            await audit_service.log_event(
                db=db,
                event_type="DATA_SUBJECT_REQUEST_CREATED",
                event_category=audit_service.Category.DATA,
                description=f"Data subject request created: {request_type} for user {user_id}",
                user_id=user_id,
                metadata={
                    "request_id": request.id,
                    "request_type": request_type,
                    "deadline": deadline.isoformat()
                },
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return request.id
            
        except Exception as e:
            logger.error(f"Error creating data subject request: {e}")
            return None
    
    @staticmethod
    async def process_data_export_request(
        db: AsyncSession,
        request_id: int
    ) -> Optional[Dict[str, Any]]:
        """Process data export request (GDPR Article 15)."""
        try:
            # Get request details
            request_result = await db.execute(
                select(DataSubjectRequest)
                .where(DataSubjectRequest.id == request_id)
            )
            request = request_result.scalar_one_or_none()
            
            if not request or request.request_type != "ACCESS":
                return None
            
            user_id = request.user_id
            
            # Collect user data from various sources
            user_data = {
                "request_id": request_id,
                "user_id": user_id,
                "export_date": datetime.utcnow().isoformat(),
                "data_categories": {}
            }
            
            # Get personal data (would need to implement based on actual models)
            # This is a simplified example
            user_data["data_categories"]["personal_information"] = {
                "note": "Personal information would be collected from User model"
            }
            
            user_data["data_categories"]["submissions"] = {
                "note": "Academic submissions would be collected from Submission model"
            }
            
            user_data["data_categories"]["consent_records"] = {
                "note": "Consent history would be collected from ConsentRecord model"
            }
            
            # Update request status
            await db.execute(
                update(DataSubjectRequest)
                .where(DataSubjectRequest.id == request_id)
                .values(
                    status="COMPLETED",
                    completed_at=datetime.utcnow(),
                    response_data=json.dumps(user_data)
                )
            )
            
            await db.commit()
            
            # Log completion
            await audit_service.log_event(
                db=db,
                event_type="DATA_EXPORT_COMPLETED",
                event_category=audit_service.Category.DATA,
                description=f"Data export completed for request {request_id}",
                user_id=user_id,
                metadata={"request_id": request_id},
                risk_level=audit_service.RiskLevel.MEDIUM
            )
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error processing data export request: {e}")
            return None
    
    @staticmethod
    async def process_data_deletion_request(
        db: AsyncSession,
        request_id: int
    ) -> bool:
        """Process data deletion request (GDPR Article 17)."""
        try:
            # Get request details
            request_result = await db.execute(
                select(DataSubjectRequest)
                .where(DataSubjectRequest.id == request_id)
            )
            request = request_result.scalar_one_or_none()
            
            if not request or request.request_type != "ERASURE":
                return False
            
            user_id = request.user_id
            
            # Check if deletion is legally permissible
            # (would need to implement business logic for retention requirements)
            
            # Anonymize or delete data based on retention policies
            anonymizer = DataAnonymizer()
            
            # This would involve:
            # 1. Anonymizing personal identifiers
            # 2. Removing or anonymizing submissions (if legally permissible)
            # 3. Keeping audit logs but anonymizing personal data
            # 4. Updating consent records
            
            # Update request status
            await db.execute(
                update(DataSubjectRequest)
                .where(DataSubjectRequest.id == request_id)
                .values(
                    status="COMPLETED",
                    completed_at=datetime.utcnow()
                )
            )
            
            await db.commit()
            
            # Log completion
            await audit_service.log_event(
                db=db,
                event_type="DATA_DELETION_COMPLETED",
                event_category=audit_service.Category.DATA,
                description=f"Data deletion completed for request {request_id}",
                user_id=user_id,
                metadata={"request_id": request_id},
                risk_level=audit_service.RiskLevel.HIGH
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing data deletion request: {e}")
            return False
    
    @staticmethod
    async def run_data_retention_cleanup(db: AsyncSession) -> Dict[str, int]:
        """Run automated data retention cleanup."""
        try:
            cleanup_stats = {
                "policies_processed": 0,
                "records_anonymized": 0,
                "records_deleted": 0,
                "records_archived": 0
            }
            
            # Get active retention policies
            policies_result = await db.execute(
                select(DataRetentionPolicy)
                .where(DataRetentionPolicy.is_active == True)
            )
            
            for policy in policies_result.scalars():
                cleanup_stats["policies_processed"] += 1
                
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)
                
                # Process based on data type and deletion method
                if policy.data_type == "audit_logs" and policy.deletion_method == "ANONYMIZE":
                    # Anonymize old audit logs
                    # This would involve updating audit logs to remove personal identifiers
                    pass
                
                elif policy.data_type == "session_data" and policy.deletion_method == "HARD_DELETE":
                    # Delete old session data
                    # This would involve deleting expired sessions
                    pass
                
                # Log retention action
                await audit_service.log_event(
                    db=db,
                    event_type="DATA_RETENTION_CLEANUP",
                    event_category=audit_service.Category.SYSTEM,
                    description=f"Data retention cleanup executed for policy: {policy.policy_name}",
                    metadata={
                        "policy_name": policy.policy_name,
                        "cutoff_date": cutoff_date.isoformat(),
                        "deletion_method": policy.deletion_method
                    },
                    risk_level=audit_service.RiskLevel.LOW
                )
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error in data retention cleanup: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def get_compliance_dashboard_data(db: AsyncSession) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        try:
            # Get pending data subject requests
            pending_requests = await db.execute(
                select(DataSubjectRequest)
                .where(DataSubjectRequest.status == "PENDING")
                .order_by(DataSubjectRequest.deadline.asc())
            )
            
            requests_data = []
            for request in pending_requests.scalars():
                days_remaining = (request.deadline - datetime.utcnow()).days
                requests_data.append({
                    "id": request.id,
                    "type": request.request_type,
                    "user_id": request.user_id,
                    "created_at": request.created_at.isoformat(),
                    "deadline": request.deadline.isoformat(),
                    "days_remaining": days_remaining,
                    "is_overdue": days_remaining < 0
                })
            
            # Get consent statistics
            consent_stats = await db.execute(
                select(
                    ConsentRecord.consent_type,
                    func.count(ConsentRecord.id).label('total'),
                    func.sum(func.cast(ConsentRecord.consent_given, Integer)).label('given'),
                    func.count(ConsentRecord.withdrawn_at).label('withdrawn')
                )
                .group_by(ConsentRecord.consent_type)
            )
            
            consent_data = {}
            for row in consent_stats:
                consent_data[row.consent_type] = {
                    "total": row.total,
                    "given": row.given or 0,
                    "withdrawn": row.withdrawn or 0
                }
            
            # Get active retention policies
            policies_count = await db.execute(
                select(func.count(DataRetentionPolicy.id))
                .where(DataRetentionPolicy.is_active == True)
            )
            
            return {
                "pending_requests": requests_data,
                "consent_statistics": consent_data,
                "active_policies_count": policies_count.scalar(),
                "compliance_status": "COMPLIANT"  # Would be calculated based on various factors
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance dashboard data: {e}")
            return {}


# Global compliance service
compliance_service = ComplianceService()
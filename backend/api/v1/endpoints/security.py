"""
Security management API endpoints.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime

from core.database import get_db
from core.auth import get_current_user, require_admin
from models.user import User
from services.security_service import security_service
from services.key_management import key_management_service
from services.compliance_service import compliance_service

router = APIRouter()


# Pydantic models
class SecurityDashboardResponse(BaseModel):
    recent_incidents: List[Dict[str, Any]]
    active_threat_indicators: int
    failed_logins_24h: int
    security_status: str


class ThreatIntelligenceRequest(BaseModel):
    indicator_type: str
    indicator_value: str
    threat_type: str
    severity: str
    source: str = "manual"
    confidence: int = 80
    expires_days: Optional[int] = None


class SecurityIncidentResponse(BaseModel):
    id: int
    incident_type: str
    severity: str
    status: str
    title: str
    created_at: datetime
    affected_user_id: Optional[int]


class KeyStatisticsResponse(BaseModel):
    active_keys: Dict[str, int]
    total_keys: int
    expiring_keys_count: int
    expiring_keys: List[Dict[str, Any]]


class ComplianceDashboardResponse(BaseModel):
    pending_requests: List[Dict[str, Any]]
    consent_statistics: Dict[str, Dict[str, int]]
    active_policies_count: int
    compliance_status: str


class DataSubjectRequestCreate(BaseModel):
    request_type: str  # ACCESS, RECTIFICATION, ERASURE, PORTABILITY, RESTRICTION
    request_details: Optional[str] = None


class ConsentRequest(BaseModel):
    consent_type: str
    purpose: str
    legal_basis: str
    consent_given: bool
    consent_method: str = "EXPLICIT"
    consent_text: Optional[str] = None


@router.get("/dashboard", response_model=SecurityDashboardResponse)
async def get_security_dashboard(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get security dashboard data."""
    try:
        dashboard_data = await security_service.get_security_dashboard_data(db)
        return SecurityDashboardResponse(**dashboard_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security dashboard data: {str(e)}"
        )


@router.post("/threat-intelligence")
async def add_threat_intelligence(
    threat_data: ThreatIntelligenceRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Add threat intelligence indicator."""
    try:
        success = await security_service.add_threat_intelligence(
            db=db,
            indicator_type=threat_data.indicator_type,
            indicator_value=threat_data.indicator_value,
            threat_type=threat_data.threat_type,
            severity=threat_data.severity,
            source=threat_data.source,
            confidence=threat_data.confidence,
            expires_days=threat_data.expires_days
        )
        
        if success:
            return {"message": "Threat intelligence added successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add threat intelligence"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add threat intelligence: {str(e)}"
        )


@router.post("/unlock-account/{user_id}")
async def unlock_user_account(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Manually unlock a user account."""
    try:
        success = await security_service.unlock_account(db, user_id, current_user.id)
        
        if success:
            return {"message": f"Account {user_id} unlocked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to unlock account"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unlock account: {str(e)}"
        )


@router.get("/intrusion-check")
async def check_intrusion_patterns(
    ip_address: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Check for intrusion patterns."""
    try:
        result = await security_service.detect_intrusion_attempt(
            db=db,
            ip_address=ip_address,
            user_id=user_id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check intrusion patterns: {str(e)}"
        )


@router.get("/keys/statistics", response_model=KeyStatisticsResponse)
async def get_key_statistics(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get encryption key statistics."""
    try:
        stats = await key_management_service.get_key_statistics(db)
        return KeyStatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get key statistics: {str(e)}"
        )


@router.post("/keys/rotate-master")
async def rotate_master_key(
    reason: str = "manual_rotation",
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Rotate master encryption key."""
    try:
        # Get current active master key
        current_key_id = await key_management_service.get_active_key(db, "master")
        
        if not current_key_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active master key found"
            )
        
        new_key_id = await key_management_service.rotate_master_key(
            db, current_key_id, reason
        )
        
        return {
            "message": "Master key rotated successfully",
            "old_key_id": current_key_id,
            "new_key_id": new_key_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rotate master key: {str(e)}"
        )


@router.post("/keys/emergency-revoke/{key_id}")
async def emergency_revoke_key(
    key_id: str,
    reason: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Emergency revocation of encryption key."""
    try:
        success = await key_management_service.emergency_key_revocation(
            db, key_id, reason
        )
        
        if success:
            return {"message": f"Key {key_id} revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to revoke key"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke key: {str(e)}"
        )


@router.get("/compliance/dashboard", response_model=ComplianceDashboardResponse)
async def get_compliance_dashboard(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get compliance dashboard data."""
    try:
        dashboard_data = await compliance_service.get_compliance_dashboard_data(db)
        return ComplianceDashboardResponse(**dashboard_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get compliance dashboard data: {str(e)}"
        )


@router.post("/compliance/data-subject-request")
async def create_data_subject_request(
    request_data: DataSubjectRequestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a data subject request (GDPR)."""
    try:
        request_id = await compliance_service.create_data_subject_request(
            db=db,
            user_id=current_user.id,
            request_type=request_data.request_type,
            request_details=request_data.request_details
        )
        
        if request_id:
            return {
                "message": "Data subject request created successfully",
                "request_id": request_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create data subject request"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create data subject request: {str(e)}"
        )


@router.post("/compliance/consent")
async def record_consent(
    consent_data: ConsentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Record user consent."""
    try:
        success = await compliance_service.record_consent(
            db=db,
            user_id=current_user.id,
            consent_type=consent_data.consent_type,
            purpose=consent_data.purpose,
            legal_basis=consent_data.legal_basis,
            consent_given=consent_data.consent_given,
            consent_method=consent_data.consent_method,
            consent_text=consent_data.consent_text
        )
        
        if success:
            return {"message": "Consent recorded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to record consent"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record consent: {str(e)}"
        )


@router.delete("/compliance/consent/{consent_type}")
async def withdraw_consent(
    consent_type: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Withdraw user consent."""
    try:
        success = await compliance_service.withdraw_consent(
            db=db,
            user_id=current_user.id,
            consent_type=consent_type
        )
        
        if success:
            return {"message": "Consent withdrawn successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to withdraw consent"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to withdraw consent: {str(e)}"
        )


@router.post("/compliance/data-retention-cleanup")
async def run_data_retention_cleanup(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Run data retention cleanup process."""
    try:
        cleanup_stats = await compliance_service.run_data_retention_cleanup(db)
        return {
            "message": "Data retention cleanup completed",
            "statistics": cleanup_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run data retention cleanup: {str(e)}"
        )


@router.get("/compliance/export-data")
async def export_user_data(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export user data (GDPR Article 15)."""
    try:
        # Create data export request
        request_id = await compliance_service.create_data_subject_request(
            db=db,
            user_id=current_user.id,
            request_type="ACCESS",
            request_details="User data export request"
        )
        
        if not request_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create export request"
            )
        
        # Process the export
        export_data = await compliance_service.process_data_export_request(db, request_id)
        
        if export_data:
            return export_data
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to export user data"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export user data: {str(e)}"
        )
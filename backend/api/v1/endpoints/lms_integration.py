"""
LMS Integration API Endpoints

Provides REST API endpoints for LMS integration functionality.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from core.auth import get_current_user, require_admin
from core.database import get_db
from models.user import User
from models.submission import Submission
from schemas.lms import (
    LMSIntegrationConfig, LMSType, AssignmentSyncRequest, AssignmentSyncResponse,
    GradeSubmissionRequest, GradeSubmissionResponse, WebhookRegistrationRequest,
    WebhookRegistrationResponse, LMSIntegrationStatus, SSOConfiguration,
    LTILaunchRequest, LTILaunchResponse, LMSWebhookEvent, LMSAssignment,
    LMSCourse, LMSEventTypes
)
from services.lms_integration_service import (
    lms_integration_service, CanvasConnector, BlackboardConnector, 
    MoodleConnector, LMSIntegrationError, LMSAuthenticationError, LMSAPIError
)
from services.auth_service import auth_service

router = APIRouter(prefix="/lms", tags=["LMS Integration"])
security = HTTPBearer()
logger = logging.getLogger(__name__)


@router.post("/configure", response_model=Dict[str, str])
async def configure_lms_integration(
    config: LMSIntegrationConfig,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Configure LMS integration for institution"""
    try:
        # Create appropriate connector based on LMS type
        if config.lms_type == LMSType.CANVAS:
            connector = CanvasConnector(config)
        elif config.lms_type == LMSType.BLACKBOARD:
            connector = BlackboardConnector(config)
        elif config.lms_type == LMSType.MOODLE:
            connector = MoodleConnector(config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported LMS type: {config.lms_type}")
        
        # Test authentication
        async with connector:
            auth_result = await connector.authenticate()
        
        # Register connector
        lms_integration_service.register_connector(config.lms_type.value, connector)
        
        # Store configuration in database (implement as needed)
        # This would typically involve saving encrypted credentials
        
        logger.info(f"LMS integration configured for {config.lms_type} at {config.institution_id}")
        
        return {
            "message": f"Successfully configured {config.lms_type} integration",
            "institution_id": config.institution_id,
            "lms_type": config.lms_type.value
        }
    
    except LMSAuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"LMS authentication failed: {str(e)}")
    except LMSIntegrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error configuring LMS integration: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{lms_type}", response_model=LMSIntegrationStatus)
async def get_integration_status(
    lms_type: LMSType,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get status of LMS integration"""
    try:
        connector = lms_integration_service.get_connector(lms_type.value)
        
        # Test connection
        connected = False
        error_message = None
        try:
            async with connector:
                await connector.authenticate()
            connected = True
        except Exception as e:
            error_message = str(e)
        
        # Get statistics (implement based on your database schema)
        total_courses = 0  # Query from database
        total_assignments = 0  # Query from database
        total_submissions = 0  # Query from database
        
        return LMSIntegrationStatus(
            lms_type=lms_type,
            institution_id=connector.config.institution_id,
            connected=connected,
            last_sync=datetime.now() if connected else None,
            total_courses=total_courses,
            total_assignments=total_assignments,
            total_submissions=total_submissions,
            webhook_configured=False,  # Check webhook status
            error_message=error_message
        )
    
    except LMSIntegrationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting integration status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/courses/{lms_type}", response_model=List[LMSCourse])
async def get_courses(
    lms_type: LMSType,
    user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get courses from LMS"""
    try:
        connector = lms_integration_service.get_connector(lms_type.value)
        
        # Use current user ID if not specified
        if not user_id:
            user_id = str(current_user.id)
        
        async with connector:
            courses = await connector.get_courses(user_id)
        
        return courses
    
    except LMSIntegrationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LMSAPIError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting courses: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/courses/{lms_type}/{course_id}/assignments", response_model=List[LMSAssignment])
async def get_assignments(
    lms_type: LMSType,
    course_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get assignments from LMS course"""
    try:
        connector = lms_integration_service.get_connector(lms_type.value)
        
        async with connector:
            assignments = await connector.get_assignments(course_id)
        
        return assignments
    
    except LMSIntegrationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LMSAPIError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting assignments: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sync/assignment", response_model=AssignmentSyncResponse)
async def sync_assignment(
    request: AssignmentSyncRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Sync assignment from LMS"""
    try:
        assignment = await lms_integration_service.sync_assignment(
            request.lms_type.value,
            request.course_id,
            request.assignment_id
        )
        
        # Store assignment in local database (implement as needed)
        # This would involve creating/updating assignment records
        
        synced_submissions = 0
        if request.auto_sync_submissions:
            # Background task to sync submissions
            background_tasks.add_task(
                sync_assignment_submissions,
                request.lms_type.value,
                request.course_id,
                request.assignment_id
            )
        
        return AssignmentSyncResponse(
            success=True,
            assignment=assignment,
            synced_submissions=synced_submissions
        )
    
    except LMSIntegrationError as e:
        return AssignmentSyncResponse(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Error syncing assignment: {str(e)}")
        return AssignmentSyncResponse(
            success=False,
            error_message="Internal server error"
        )


@router.post("/grade/submit", response_model=GradeSubmissionResponse)
async def submit_grade(
    request: GradeSubmissionRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Submit verification result as grade to LMS"""
    try:
        # Get submission from database
        submission = db.query(Submission).filter(
            Submission.id == request.submission_id
        ).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Submit grade to LMS
        success = await lms_integration_service.submit_verification_result(
            request.lms_type.value,
            request.assignment_id,
            request.user_id,
            submission
        )
        
        if success:
            # Update submission status in database
            submission.lms_grade_submitted = True
            submission.lms_grade_submitted_at = datetime.now()
            db.commit()
        
        return GradeSubmissionResponse(
            success=success,
            error_message=None if success else "Failed to submit grade to LMS"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting grade: {str(e)}")
        return GradeSubmissionResponse(
            success=False,
            error_message="Internal server error"
        )


@router.post("/webhook/register", response_model=WebhookRegistrationResponse)
async def register_webhook(
    request: WebhookRegistrationRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Register webhook with LMS"""
    try:
        connector = lms_integration_service.get_connector(request.lms_type.value)
        
        async with connector:
            webhook_result = await connector.create_webhook(
                request.webhook_url,
                request.events
            )
        
        return WebhookRegistrationResponse(
            success=True,
            webhook_id=webhook_result.get("id"),
            webhook_url=request.webhook_url,
            events=request.events,
            setup_required=webhook_result.get("setup_required", False),
            instructions=webhook_result.get("instructions")
        )
    
    except LMSIntegrationError as e:
        return WebhookRegistrationResponse(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Error registering webhook: {str(e)}")
        return WebhookRegistrationResponse(
            success=False,
            error_message="Internal server error"
        )


@router.post("/webhook/{lms_type}")
async def handle_webhook(
    lms_type: LMSType,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Handle incoming webhook from LMS"""
    try:
        # Get request body
        body = await request.body()
        headers = dict(request.headers)
        
        # Verify webhook signature (implement based on LMS requirements)
        # This would typically involve HMAC verification
        
        # Parse webhook data
        event_data = await request.json()
        
        # Add LMS type and process in background
        event_data["lms_type"] = lms_type.value
        
        background_tasks.add_task(
            process_webhook_event,
            lms_type.value,
            event_data
        )
        
        return {"status": "received"}
    
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid webhook data")


@router.post("/sso/configure")
async def configure_sso(
    config: SSOConfiguration,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Configure SSO integration with LMS"""
    try:
        # Store SSO configuration (implement as needed)
        # This would involve saving SAML/OAuth configuration
        
        logger.info(f"SSO configured for {config.lms_type}")
        
        return {
            "message": f"SSO configuration saved for {config.lms_type}",
            "sso_enabled": config.sso_enabled
        }
    
    except Exception as e:
        logger.error(f"Error configuring SSO: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/lti/launch", response_model=LTILaunchResponse)
async def handle_lti_launch(
    launch_data: LTILaunchRequest,
    db: Session = Depends(get_db)
):
    """Handle LTI launch from LMS"""
    try:
        # Validate LTI signature (implement OAuth signature validation)
        
        # Authenticate or create user based on LTI data
        user = await auth_service.authenticate_lti_user(launch_data, db)
        
        # Generate session token
        access_token = auth_service.create_access_token(data={"sub": str(user.id)})
        
        # Determine redirect URL based on context
        redirect_url = f"/dashboard?token={access_token}"
        if launch_data.custom_canvas_assignment_id:
            redirect_url += f"&assignment_id={launch_data.custom_canvas_assignment_id}"
        
        return LTILaunchResponse(
            success=True,
            redirect_url=redirect_url,
            user_authenticated=True,
            assignment_id=launch_data.custom_canvas_assignment_id,
            course_id=launch_data.custom_canvas_course_id
        )
    
    except Exception as e:
        logger.error(f"Error handling LTI launch: {str(e)}")
        return LTILaunchResponse(
            success=False,
            error_message="LTI launch failed"
        )


# Background task functions

async def sync_assignment_submissions(lms_type: str, course_id: str, assignment_id: str):
    """Background task to sync assignment submissions"""
    try:
        connector = lms_integration_service.get_connector(lms_type)
        # Implement submission sync logic
        logger.info(f"Syncing submissions for assignment {assignment_id}")
    except Exception as e:
        logger.error(f"Error syncing submissions: {str(e)}")


async def process_webhook_event(lms_type: str, event_data: Dict):
    """Background task to process webhook events"""
    try:
        success = await lms_integration_service.handle_webhook_event(lms_type, event_data)
        if success:
            logger.info(f"Processed webhook event: {event_data.get('event_type')}")
        else:
            logger.warning(f"Failed to process webhook event: {event_data.get('event_type')}")
    except Exception as e:
        logger.error(f"Error processing webhook event: {str(e)}")


# Register webhook event handlers
def setup_webhook_handlers():
    """Setup webhook event handlers"""
    
    async def handle_assignment_created(event: LMSWebhookEvent):
        """Handle assignment created event"""
        logger.info(f"Assignment created: {event.assignment_id}")
        # Implement assignment creation logic
    
    async def handle_submission_created(event: LMSWebhookEvent):
        """Handle submission created event"""
        logger.info(f"Submission created: {event.submission_id}")
        # Implement automatic verification trigger
    
    async def handle_grade_changed(event: LMSWebhookEvent):
        """Handle grade changed event"""
        logger.info(f"Grade changed for assignment: {event.assignment_id}")
        # Implement grade sync logic
    
    # Register handlers
    lms_integration_service.register_webhook_handler(
        LMSEventTypes.ASSIGNMENT_CREATED, 
        handle_assignment_created
    )
    lms_integration_service.register_webhook_handler(
        LMSEventTypes.SUBMISSION_CREATED, 
        handle_submission_created
    )
    lms_integration_service.register_webhook_handler(
        LMSEventTypes.GRADE_CHANGED, 
        handle_grade_changed
    )


# Initialize webhook handlers
setup_webhook_handlers()
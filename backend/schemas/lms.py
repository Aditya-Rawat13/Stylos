"""
LMS Integration Schemas

Pydantic models for LMS integration data structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class LMSType(str, Enum):
    """Supported LMS types"""
    CANVAS = "canvas"
    BLACKBOARD = "blackboard"
    MOODLE = "moodle"


class LMSIntegrationConfig(BaseModel):
    """Configuration for LMS integration"""
    lms_type: LMSType
    base_url: str = Field(..., description="Base URL of the LMS instance")
    api_key: Optional[str] = Field(None, description="API key or token")
    client_id: Optional[str] = Field(None, description="OAuth client ID")
    client_secret: Optional[str] = Field(None, description="OAuth client secret")
    institution_id: str = Field(..., description="Institution identifier")
    enabled: bool = Field(True, description="Whether integration is enabled")
    webhook_secret: Optional[str] = Field(None, description="Webhook verification secret")
    
    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Base URL must start with http:// or https://')
        return v.rstrip('/')


class LMSUser(BaseModel):
    """LMS user representation"""
    id: str
    name: str
    email: str
    login_id: Optional[str] = None
    sis_user_id: Optional[str] = None
    role: Optional[str] = None


class LMSCourse(BaseModel):
    """LMS course representation"""
    id: str
    name: str
    code: str
    term: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    enrollment_term_id: Optional[str] = None
    sis_course_id: Optional[str] = None
    workflow_state: Optional[str] = None


class LMSAssignment(BaseModel):
    """LMS assignment representation"""
    id: str
    course_id: str
    name: str
    description: Optional[str] = None
    due_date: Optional[str] = None
    unlock_date: Optional[str] = None
    lock_date: Optional[str] = None
    points_possible: float = 0
    submission_types: List[str] = []
    allowed_extensions: List[str] = []
    published: bool = True
    workflow_state: Optional[str] = None
    
    @validator('submission_types')
    def validate_submission_types(cls, v):
        valid_types = [
            'online_text_entry', 'online_upload', 'online_url', 
            'media_recording', 'external_tool', 'none'
        ]
        for submission_type in v:
            if submission_type not in valid_types:
                raise ValueError(f'Invalid submission type: {submission_type}')
        return v


class LMSSubmission(BaseModel):
    """LMS submission representation"""
    id: str
    assignment_id: str
    user_id: str
    submitted_at: Optional[datetime] = None
    grade: Optional[Union[str, float]] = None
    score: Optional[float] = None
    workflow_state: Optional[str] = None
    submission_type: Optional[str] = None
    body: Optional[str] = None
    url: Optional[str] = None
    attachments: List[Dict[str, Any]] = []


class LMSGrade(BaseModel):
    """Grade to be submitted to LMS"""
    assignment_id: str
    user_id: str
    course_id: str
    score: float = Field(..., ge=0, le=100, description="Grade score (0-100)")
    comment: Optional[str] = Field(None, max_length=1000, description="Feedback comment")
    submitted_at: datetime = Field(default_factory=datetime.now)
    
    @validator('score')
    def validate_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Score must be between 0 and 100')
        return round(v, 2)


class LMSWebhookEvent(BaseModel):
    """Webhook event from LMS"""
    event_type: str = Field(..., description="Type of event (e.g., assignment_created, submission_created)")
    event_time: datetime = Field(default_factory=datetime.now)
    lms_type: LMSType
    institution_id: str
    user_id: Optional[str] = None
    course_id: Optional[str] = None
    assignment_id: Optional[str] = None
    submission_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")


class AssignmentSyncRequest(BaseModel):
    """Request to sync assignment from LMS"""
    lms_type: LMSType
    course_id: str
    assignment_id: str
    auto_sync_submissions: bool = Field(False, description="Whether to automatically sync submissions")


class AssignmentSyncResponse(BaseModel):
    """Response from assignment sync"""
    success: bool
    assignment: Optional[LMSAssignment] = None
    synced_submissions: int = 0
    error_message: Optional[str] = None


class GradeSubmissionRequest(BaseModel):
    """Request to submit grade back to LMS"""
    lms_type: LMSType
    assignment_id: str
    user_id: str
    submission_id: str
    include_verification_details: bool = Field(True, description="Include detailed verification results in comment")


class GradeSubmissionResponse(BaseModel):
    """Response from grade submission"""
    success: bool
    grade_submitted: Optional[LMSGrade] = None
    lms_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class WebhookRegistrationRequest(BaseModel):
    """Request to register webhook with LMS"""
    lms_type: LMSType
    webhook_url: str
    events: List[str] = Field(default_factory=lambda: ["assignment_created", "submission_created"])
    description: Optional[str] = "Project Stylos Integration Webhook"


class WebhookRegistrationResponse(BaseModel):
    """Response from webhook registration"""
    success: bool
    webhook_id: Optional[str] = None
    webhook_url: Optional[str] = None
    events: List[str] = []
    setup_required: bool = False
    instructions: Optional[str] = None
    error_message: Optional[str] = None


class LMSIntegrationStatus(BaseModel):
    """Status of LMS integration"""
    lms_type: LMSType
    institution_id: str
    connected: bool
    last_sync: Optional[datetime] = None
    total_courses: int = 0
    total_assignments: int = 0
    total_submissions: int = 0
    webhook_configured: bool = False
    error_message: Optional[str] = None


class SSOConfiguration(BaseModel):
    """SSO configuration for LMS integration"""
    lms_type: LMSType
    sso_enabled: bool = False
    saml_metadata_url: Optional[str] = None
    saml_entity_id: Optional[str] = None
    oauth_authorize_url: Optional[str] = None
    oauth_token_url: Optional[str] = None
    lti_consumer_key: Optional[str] = None
    lti_shared_secret: Optional[str] = None
    
    @validator('saml_metadata_url', 'oauth_authorize_url', 'oauth_token_url')
    def validate_urls(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class LTILaunchRequest(BaseModel):
    """LTI launch request data"""
    lti_message_type: str = "basic-lti-launch-request"
    lti_version: str = "LTI-1p0"
    resource_link_id: str
    user_id: str
    roles: str
    context_id: str
    context_title: Optional[str] = None
    launch_presentation_return_url: Optional[str] = None
    tool_consumer_instance_guid: str
    custom_canvas_assignment_id: Optional[str] = None
    custom_canvas_course_id: Optional[str] = None


class LTILaunchResponse(BaseModel):
    """Response to LTI launch"""
    success: bool
    redirect_url: Optional[str] = None
    user_authenticated: bool = False
    assignment_id: Optional[str] = None
    course_id: Optional[str] = None
    error_message: Optional[str] = None


# Event type constants
class LMSEventTypes:
    """Constants for LMS webhook event types"""
    ASSIGNMENT_CREATED = "assignment_created"
    ASSIGNMENT_UPDATED = "assignment_updated"
    ASSIGNMENT_DELETED = "assignment_deleted"
    SUBMISSION_CREATED = "submission_created"
    SUBMISSION_UPDATED = "submission_updated"
    GRADE_CHANGED = "grade_changed"
    COURSE_CREATED = "course_created"
    COURSE_UPDATED = "course_updated"
    USER_ENROLLED = "user_enrolled"
    USER_UNENROLLED = "user_unenrolled"


# Submission type constants
class LMSSubmissionTypes:
    """Constants for LMS submission types"""
    ONLINE_TEXT = "online_text_entry"
    ONLINE_UPLOAD = "online_upload"
    ONLINE_URL = "online_url"
    MEDIA_RECORDING = "media_recording"
    EXTERNAL_TOOL = "external_tool"
    NONE = "none"
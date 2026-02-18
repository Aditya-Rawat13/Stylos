"""
LMS Integration Configuration

Configuration settings and utilities for LMS integrations.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field

from schemas.lms import LMSType


class LMSIntegrationSettings(BaseSettings):
    """LMS integration configuration settings"""
    
    # General settings
    LMS_INTEGRATION_ENABLED: bool = Field(True, env="LMS_INTEGRATION_ENABLED")
    LMS_WEBHOOK_BASE_URL: str = Field("", env="LMS_WEBHOOK_BASE_URL")
    LMS_WEBHOOK_SECRET: str = Field("", env="LMS_WEBHOOK_SECRET")
    
    # Canvas settings
    CANVAS_ENABLED: bool = Field(True, env="CANVAS_ENABLED")
    CANVAS_API_TIMEOUT: int = Field(30, env="CANVAS_API_TIMEOUT")
    CANVAS_MAX_RETRIES: int = Field(3, env="CANVAS_MAX_RETRIES")
    
    # Blackboard settings
    BLACKBOARD_ENABLED: bool = Field(True, env="BLACKBOARD_ENABLED")
    BLACKBOARD_API_TIMEOUT: int = Field(30, env="BLACKBOARD_API_TIMEOUT")
    BLACKBOARD_MAX_RETRIES: int = Field(3, env="BLACKBOARD_MAX_RETRIES")
    
    # Moodle settings
    MOODLE_ENABLED: bool = Field(True, env="MOODLE_ENABLED")
    MOODLE_API_TIMEOUT: int = Field(30, env="MOODLE_API_TIMEOUT")
    MOODLE_MAX_RETRIES: int = Field(3, env="MOODLE_MAX_RETRIES")
    
    # SSO settings
    SSO_ENABLED: bool = Field(True, env="SSO_ENABLED")
    SAML_ENABLED: bool = Field(True, env="SAML_ENABLED")
    OAUTH_ENABLED: bool = Field(True, env="OAUTH_ENABLED")
    LTI_ENABLED: bool = Field(True, env="LTI_ENABLED")
    
    # Webhook settings
    WEBHOOK_RETRY_ATTEMPTS: int = Field(3, env="WEBHOOK_RETRY_ATTEMPTS")
    WEBHOOK_RETRY_DELAY: int = Field(5, env="WEBHOOK_RETRY_DELAY")  # seconds
    WEBHOOK_TIMEOUT: int = Field(30, env="WEBHOOK_TIMEOUT")
    
    # Grade passback settings
    GRADE_PASSBACK_ENABLED: bool = Field(True, env="GRADE_PASSBACK_ENABLED")
    GRADE_PASSBACK_AUTO: bool = Field(False, env="GRADE_PASSBACK_AUTO")
    GRADE_PASSBACK_THRESHOLD: float = Field(70.0, env="GRADE_PASSBACK_THRESHOLD")
    
    # Sync settings
    AUTO_SYNC_ENABLED: bool = Field(True, env="AUTO_SYNC_ENABLED")
    SYNC_INTERVAL_HOURS: int = Field(24, env="SYNC_INTERVAL_HOURS")
    SYNC_BATCH_SIZE: int = Field(100, env="SYNC_BATCH_SIZE")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class LMSEndpoints:
    """LMS API endpoint configurations"""
    
    CANVAS_ENDPOINTS = {
        "auth": "/api/v1/users/self",
        "courses": "/api/v1/courses",
        "assignments": "/api/v1/courses/{course_id}/assignments",
        "submissions": "/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions",
        "grades": "/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{user_id}",
        "webhooks": "/api/lti/subscriptions",
        "users": "/api/v1/users/{user_id}"
    }
    
    BLACKBOARD_ENDPOINTS = {
        "auth": "/learn/api/public/v1/oauth2/token",
        "courses": "/learn/api/public/v1/users/{user_id}/courses",
        "assignments": "/learn/api/public/v1/courses/{course_id}/gradebook/columns",
        "submissions": "/learn/api/public/v1/courses/{course_id}/gradebook/columns/{column_id}/users/{user_id}",
        "grades": "/learn/api/public/v1/courses/{course_id}/gradebook/columns/{column_id}/users/{user_id}",
        "webhooks": "/learn/api/public/v1/webhooks",
        "users": "/learn/api/public/v1/users/{user_id}"
    }
    
    MOODLE_ENDPOINTS = {
        "auth": "/webservice/rest/server.php?wsfunction=core_webservice_get_site_info",
        "courses": "/webservice/rest/server.php?wsfunction=core_enrol_get_users_courses",
        "assignments": "/webservice/rest/server.php?wsfunction=mod_assign_get_assignments",
        "submissions": "/webservice/rest/server.php?wsfunction=mod_assign_get_submissions",
        "grades": "/webservice/rest/server.php?wsfunction=mod_assign_save_grade",
        "users": "/webservice/rest/server.php?wsfunction=core_user_get_users"
    }
    
    @classmethod
    def get_endpoints(cls, lms_type: LMSType) -> Dict[str, str]:
        """Get endpoints for specific LMS type"""
        if lms_type == LMSType.CANVAS:
            return cls.CANVAS_ENDPOINTS
        elif lms_type == LMSType.BLACKBOARD:
            return cls.BLACKBOARD_ENDPOINTS
        elif lms_type == LMSType.MOODLE:
            return cls.MOODLE_ENDPOINTS
        else:
            raise ValueError(f"Unsupported LMS type: {lms_type}")


class LMSEventMapping:
    """Mapping of LMS-specific events to standardized events"""
    
    CANVAS_EVENT_MAPPING = {
        "assignment_created": "assignment_created",
        "assignment_updated": "assignment_updated",
        "submission_created": "submission_created",
        "submission_updated": "submission_updated",
        "grade_change": "grade_changed",
        "enrollment_created": "user_enrolled",
        "enrollment_deleted": "user_unenrolled"
    }
    
    BLACKBOARD_EVENT_MAPPING = {
        "course.assignment.created": "assignment_created",
        "course.assignment.modified": "assignment_updated",
        "course.assignment.deleted": "assignment_deleted",
        "course.grade.created": "grade_changed",
        "course.grade.modified": "grade_changed",
        "course.membership.created": "user_enrolled",
        "course.membership.deleted": "user_unenrolled"
    }
    
    MOODLE_EVENT_MAPPING = {
        "\\mod_assign\\event\\assessable_submitted": "submission_created",
        "\\core\\event\\user_graded": "grade_changed",
        "\\core\\event\\user_enrolment_created": "user_enrolled",
        "\\core\\event\\user_enrolment_deleted": "user_unenrolled"
    }
    
    @classmethod
    def get_standardized_event(cls, lms_type: LMSType, lms_event: str) -> Optional[str]:
        """Get standardized event name from LMS-specific event"""
        if lms_type == LMSType.CANVAS:
            return cls.CANVAS_EVENT_MAPPING.get(lms_event)
        elif lms_type == LMSType.BLACKBOARD:
            return cls.BLACKBOARD_EVENT_MAPPING.get(lms_event)
        elif lms_type == LMSType.MOODLE:
            return cls.MOODLE_EVENT_MAPPING.get(lms_event)
        return None


class LMSGradeCalculator:
    """Grade calculation utilities for LMS integration"""
    
    @staticmethod
    def calculate_grade_from_verification(
        authorship_score: float,
        ai_probability: float,
        duplicate_matches: int,
        base_points: float = 100.0
    ) -> Dict[str, float]:
        """Calculate grade based on verification results"""
        
        # Start with base points
        final_score = base_points
        
        # Authorship penalty (if score below 70%)
        if authorship_score < 70:
            authorship_penalty = (70 - authorship_score) * 0.5
            final_score -= authorship_penalty
        
        # AI detection penalty
        ai_penalty = ai_probability * 30  # Up to 30 point penalty for 100% AI
        final_score -= ai_penalty
        
        # Duplicate penalty
        if duplicate_matches > 0:
            duplicate_penalty = min(duplicate_matches * 10, 50)  # Max 50 point penalty
            final_score -= duplicate_penalty
        
        # Ensure score is not negative
        final_score = max(0, final_score)
        
        return {
            "final_score": round(final_score, 2),
            "authorship_penalty": round((70 - authorship_score) * 0.5 if authorship_score < 70 else 0, 2),
            "ai_penalty": round(ai_penalty, 2),
            "duplicate_penalty": round(min(duplicate_matches * 10, 50) if duplicate_matches > 0 else 0, 2)
        }
    
    @staticmethod
    def generate_grade_comment(
        authorship_score: float,
        ai_probability: float,
        duplicate_matches: int,
        overall_status: str
    ) -> str:
        """Generate grade comment based on verification results"""
        
        comments = []
        
        # Overall status
        comments.append(f"Verification Status: {overall_status}")
        
        # Authorship analysis
        if authorship_score >= 85:
            comments.append(f"Authorship: Excellent match ({authorship_score:.1f}%)")
        elif authorship_score >= 70:
            comments.append(f"Authorship: Good match ({authorship_score:.1f}%)")
        else:
            comments.append(f"Authorship: Concerns detected ({authorship_score:.1f}%)")
        
        # AI detection
        if ai_probability <= 0.2:
            comments.append(f"AI Detection: Human-written ({(1-ai_probability)*100:.1f}% confidence)")
        elif ai_probability <= 0.5:
            comments.append(f"AI Detection: Possibly AI-assisted ({ai_probability*100:.1f}% AI probability)")
        else:
            comments.append(f"AI Detection: Likely AI-generated ({ai_probability*100:.1f}% AI probability)")
        
        # Duplicate detection
        if duplicate_matches == 0:
            comments.append("Duplicate Detection: No matches found")
        else:
            comments.append(f"Duplicate Detection: {duplicate_matches} similar submission(s) found")
        
        return " | ".join(comments)


class LMSFeatureFlags:
    """Feature flags for LMS integration capabilities"""
    
    def __init__(self, settings: LMSIntegrationSettings):
        self.settings = settings
    
    def is_lms_enabled(self, lms_type: LMSType) -> bool:
        """Check if specific LMS type is enabled"""
        if not self.settings.LMS_INTEGRATION_ENABLED:
            return False
        
        if lms_type == LMSType.CANVAS:
            return self.settings.CANVAS_ENABLED
        elif lms_type == LMSType.BLACKBOARD:
            return self.settings.BLACKBOARD_ENABLED
        elif lms_type == LMSType.MOODLE:
            return self.settings.MOODLE_ENABLED
        
        return False
    
    def is_sso_enabled(self) -> bool:
        """Check if SSO is enabled"""
        return self.settings.SSO_ENABLED
    
    def is_webhook_enabled(self) -> bool:
        """Check if webhooks are enabled"""
        return bool(self.settings.LMS_WEBHOOK_BASE_URL)
    
    def is_grade_passback_enabled(self) -> bool:
        """Check if grade passback is enabled"""
        return self.settings.GRADE_PASSBACK_ENABLED
    
    def is_auto_sync_enabled(self) -> bool:
        """Check if auto sync is enabled"""
        return self.settings.AUTO_SYNC_ENABLED


# Global settings instance
lms_settings = LMSIntegrationSettings()
lms_features = LMSFeatureFlags(lms_settings)
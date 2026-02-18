"""
LMS Integration Service

Provides integration capabilities with major Learning Management Systems
including Canvas, Blackboard, and Moodle.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import aiohttp
from pydantic import BaseModel, Field

from core.config import settings
from models.submission import Submission
from schemas.lms import (
    LMSAssignment, LMSCourse, LMSUser, LMSGrade,
    LMSWebhookEvent, LMSIntegrationConfig
)

logger = logging.getLogger(__name__)


class LMSIntegrationError(Exception):
    """Base exception for LMS integration errors"""
    pass


class LMSAuthenticationError(LMSIntegrationError):
    """Authentication failed with LMS"""
    pass


class LMSAPIError(LMSIntegrationError):
    """API call to LMS failed"""
    pass


class BaseLMSConnector(ABC):
    """Abstract base class for LMS connectors"""
    
    def __init__(self, config: LMSIntegrationConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.client_id = config.client_id
        self.client_secret = config.client_secret
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    @abstractmethod
    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate with the LMS"""
        pass
    
    @abstractmethod
    async def get_courses(self, user_id: str) -> List[LMSCourse]:
        """Get courses for a user"""
        pass
    
    @abstractmethod
    async def get_assignments(self, course_id: str) -> List[LMSAssignment]:
        """Get assignments for a course"""
        pass
    
    @abstractmethod
    async def submit_grade(self, assignment_id: str, user_id: str, grade: LMSGrade) -> bool:
        """Submit grade back to LMS"""
        pass
    
    @abstractmethod
    async def create_webhook(self, webhook_url: str, events: List[str]) -> Dict[str, Any]:
        """Create webhook subscription"""
        pass


class CanvasConnector(BaseLMSConnector):
    """Canvas LMS connector"""
    
    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate using Canvas API token"""
        if not self.api_key:
            raise LMSAuthenticationError("Canvas API key not provided")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with self._session.get(
            f"{self.base_url}/api/v1/users/self",
            headers=headers
        ) as response:
            if response.status == 200:
                user_data = await response.json()
                logger.info(f"Canvas authentication successful for user: {user_data.get('name')}")
                return user_data
            else:
                raise LMSAuthenticationError(f"Canvas authentication failed: {response.status}")
    
    async def get_courses(self, user_id: str) -> List[LMSCourse]:
        """Get Canvas courses for user"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with self._session.get(
            f"{self.base_url}/api/v1/courses",
            headers=headers,
            params={"enrollment_state": "active", "per_page": 100}
        ) as response:
            if response.status == 200:
                courses_data = await response.json()
                return [
                    LMSCourse(
                        id=str(course["id"]),
                        name=course["name"],
                        code=course.get("course_code", ""),
                        term=course.get("term", {}).get("name", ""),
                        start_date=course.get("start_at"),
                        end_date=course.get("end_at")
                    )
                    for course in courses_data
                ]
            else:
                raise LMSAPIError(f"Failed to get Canvas courses: {response.status}")
    
    async def get_assignments(self, course_id: str) -> List[LMSAssignment]:
        """Get Canvas assignments for course"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with self._session.get(
            f"{self.base_url}/api/v1/courses/{course_id}/assignments",
            headers=headers,
            params={"per_page": 100}
        ) as response:
            if response.status == 200:
                assignments_data = await response.json()
                return [
                    LMSAssignment(
                        id=str(assignment["id"]),
                        course_id=course_id,
                        name=assignment["name"],
                        description=assignment.get("description", ""),
                        due_date=assignment.get("due_at"),
                        points_possible=assignment.get("points_possible", 0),
                        submission_types=assignment.get("submission_types", [])
                    )
                    for assignment in assignments_data
                ]
            else:
                raise LMSAPIError(f"Failed to get Canvas assignments: {response.status}")
    
    async def submit_grade(self, assignment_id: str, user_id: str, grade: LMSGrade) -> bool:
        """Submit grade to Canvas"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        grade_data = {
            "submission": {
                "posted_grade": grade.score,
                "text_comment": grade.comment
            }
        }
        
        async with self._session.put(
            f"{self.base_url}/api/v1/courses/{grade.course_id}/assignments/{assignment_id}/submissions/{user_id}",
            headers=headers,
            json=grade_data
        ) as response:
            if response.status in [200, 201]:
                logger.info(f"Grade submitted to Canvas: {grade.score} for user {user_id}")
                return True
            else:
                logger.error(f"Failed to submit grade to Canvas: {response.status}")
                return False
    
    async def create_webhook(self, webhook_url: str, events: List[str]) -> Dict[str, Any]:
        """Create Canvas webhook subscription"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        webhook_data = {
            "subscription": {
                "notification_url": webhook_url,
                "transport_metadata": {"url": webhook_url},
                "transport_type": "https"
            }
        }
        
        async with self._session.post(
            f"{self.base_url}/api/lti/subscriptions",
            headers=headers,
            json=webhook_data
        ) as response:
            if response.status in [200, 201]:
                return await response.json()
            else:
                raise LMSAPIError(f"Failed to create Canvas webhook: {response.status}")


class BlackboardConnector(BaseLMSConnector):
    """Blackboard Learn connector"""
    
    def __init__(self, config: LMSIntegrationConfig):
        super().__init__(config)
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
    
    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate using Blackboard REST API"""
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        async with self._session.post(
            f"{self.base_url}/learn/api/public/v1/oauth2/token",
            data=auth_data
        ) as response:
            if response.status == 200:
                token_data = await response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expires = datetime.now() + timedelta(seconds=expires_in)
                logger.info("Blackboard authentication successful")
                return token_data
            else:
                raise LMSAuthenticationError(f"Blackboard authentication failed: {response.status}")
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid access token"""
        if not self._access_token or (self._token_expires and datetime.now() >= self._token_expires):
            await self.authenticate()
    
    async def get_courses(self, user_id: str) -> List[LMSCourse]:
        """Get Blackboard courses for user"""
        await self._ensure_authenticated()
        
        headers = {"Authorization": f"Bearer {self._access_token}"}
        
        async with self._session.get(
            f"{self.base_url}/learn/api/public/v1/users/{user_id}/courses",
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                courses_data = data.get("results", [])
                return [
                    LMSCourse(
                        id=course["id"],
                        name=course["name"],
                        code=course.get("courseId", ""),
                        term=course.get("termId", ""),
                        start_date=course.get("availability", {}).get("available", {}).get("from"),
                        end_date=course.get("availability", {}).get("available", {}).get("until")
                    )
                    for course in courses_data
                ]
            else:
                raise LMSAPIError(f"Failed to get Blackboard courses: {response.status}")
    
    async def get_assignments(self, course_id: str) -> List[LMSAssignment]:
        """Get Blackboard assignments for course"""
        await self._ensure_authenticated()
        
        headers = {"Authorization": f"Bearer {self._access_token}"}
        
        async with self._session.get(
            f"{self.base_url}/learn/api/public/v1/courses/{course_id}/gradebook/columns",
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                assignments_data = data.get("results", [])
                return [
                    LMSAssignment(
                        id=assignment["id"],
                        course_id=course_id,
                        name=assignment["name"],
                        description=assignment.get("description", ""),
                        due_date=assignment.get("dueDate"),
                        points_possible=assignment.get("score", {}).get("possible", 0),
                        submission_types=["online_text_entry"]  # Default for Blackboard
                    )
                    for assignment in assignments_data
                    if assignment.get("contentHandler", {}).get("id") == "resource/x-bb-assignment"
                ]
            else:
                raise LMSAPIError(f"Failed to get Blackboard assignments: {response.status}")
    
    async def submit_grade(self, assignment_id: str, user_id: str, grade: LMSGrade) -> bool:
        """Submit grade to Blackboard"""
        await self._ensure_authenticated()
        
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        
        grade_data = {
            "score": grade.score,
            "text": grade.comment,
            "feedback": grade.comment
        }
        
        async with self._session.patch(
            f"{self.base_url}/learn/api/public/v1/courses/{grade.course_id}/gradebook/columns/{assignment_id}/users/{user_id}",
            headers=headers,
            json=grade_data
        ) as response:
            if response.status in [200, 201]:
                logger.info(f"Grade submitted to Blackboard: {grade.score} for user {user_id}")
                return True
            else:
                logger.error(f"Failed to submit grade to Blackboard: {response.status}")
                return False
    
    async def create_webhook(self, webhook_url: str, events: List[str]) -> Dict[str, Any]:
        """Create Blackboard webhook subscription"""
        await self._ensure_authenticated()
        
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        
        webhook_data = {
            "url": webhook_url,
            "events": events,
            "description": "Project Stylos Integration"
        }
        
        async with self._session.post(
            f"{self.base_url}/learn/api/public/v1/webhooks",
            headers=headers,
            json=webhook_data
        ) as response:
            if response.status in [200, 201]:
                return await response.json()
            else:
                raise LMSAPIError(f"Failed to create Blackboard webhook: {response.status}")


class MoodleConnector(BaseLMSConnector):
    """Moodle connector using Web Services API"""
    
    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate using Moodle token"""
        if not self.api_key:
            raise LMSAuthenticationError("Moodle token not provided")
        
        # Test authentication by getting site info
        params = {
            "wstoken": self.api_key,
            "wsfunction": "core_webservice_get_site_info",
            "moodlewsrestformat": "json"
        }
        
        async with self._session.get(
            f"{self.base_url}/webservice/rest/server.php",
            params=params
        ) as response:
            if response.status == 200:
                site_info = await response.json()
                if "exception" in site_info:
                    raise LMSAuthenticationError(f"Moodle authentication failed: {site_info['message']}")
                logger.info(f"Moodle authentication successful for site: {site_info.get('sitename')}")
                return site_info
            else:
                raise LMSAuthenticationError(f"Moodle authentication failed: {response.status}")
    
    async def get_courses(self, user_id: str) -> List[LMSCourse]:
        """Get Moodle courses for user"""
        params = {
            "wstoken": self.api_key,
            "wsfunction": "core_enrol_get_users_courses",
            "moodlewsrestformat": "json",
            "userid": user_id
        }
        
        async with self._session.get(
            f"{self.base_url}/webservice/rest/server.php",
            params=params
        ) as response:
            if response.status == 200:
                courses_data = await response.json()
                if isinstance(courses_data, dict) and "exception" in courses_data:
                    raise LMSAPIError(f"Moodle API error: {courses_data['message']}")
                
                return [
                    LMSCourse(
                        id=str(course["id"]),
                        name=course["fullname"],
                        code=course.get("shortname", ""),
                        term=course.get("category", ""),
                        start_date=datetime.fromtimestamp(course["startdate"]).isoformat() if course.get("startdate") else None,
                        end_date=datetime.fromtimestamp(course["enddate"]).isoformat() if course.get("enddate") else None
                    )
                    for course in courses_data
                ]
            else:
                raise LMSAPIError(f"Failed to get Moodle courses: {response.status}")
    
    async def get_assignments(self, course_id: str) -> List[LMSAssignment]:
        """Get Moodle assignments for course"""
        params = {
            "wstoken": self.api_key,
            "wsfunction": "mod_assign_get_assignments",
            "moodlewsrestformat": "json",
            "courseids[0]": course_id
        }
        
        async with self._session.get(
            f"{self.base_url}/webservice/rest/server.php",
            params=params
        ) as response:
            if response.status == 200:
                data = await response.json()
                if isinstance(data, dict) and "exception" in data:
                    raise LMSAPIError(f"Moodle API error: {data['message']}")
                
                assignments_data = data.get("courses", [{}])[0].get("assignments", [])
                return [
                    LMSAssignment(
                        id=str(assignment["id"]),
                        course_id=course_id,
                        name=assignment["name"],
                        description=assignment.get("intro", ""),
                        due_date=datetime.fromtimestamp(assignment["duedate"]).isoformat() if assignment.get("duedate") else None,
                        points_possible=assignment.get("grade", 0),
                        submission_types=["online_text"]  # Default for Moodle
                    )
                    for assignment in assignments_data
                ]
            else:
                raise LMSAPIError(f"Failed to get Moodle assignments: {response.status}")
    
    async def submit_grade(self, assignment_id: str, user_id: str, grade: LMSGrade) -> bool:
        """Submit grade to Moodle"""
        params = {
            "wstoken": self.api_key,
            "wsfunction": "mod_assign_save_grade",
            "moodlewsrestformat": "json",
            "assignmentid": assignment_id,
            "userid": user_id,
            "grade": grade.score,
            "attemptnumber": -1,  # Latest attempt
            "addattempt": 0,
            "workflowstate": "",
            "applytoall": 0
        }
        
        if grade.comment:
            params["plugindata[assignfeedbackcomments_editor][text]"] = grade.comment
            params["plugindata[assignfeedbackcomments_editor][format]"] = 1
        
        async with self._session.post(
            f"{self.base_url}/webservice/rest/server.php",
            data=params
        ) as response:
            if response.status == 200:
                result = await response.json()
                if isinstance(result, dict) and "exception" in result:
                    logger.error(f"Failed to submit grade to Moodle: {result['message']}")
                    return False
                logger.info(f"Grade submitted to Moodle: {grade.score} for user {user_id}")
                return True
            else:
                logger.error(f"Failed to submit grade to Moodle: {response.status}")
                return False
    
    async def create_webhook(self, webhook_url: str, events: List[str]) -> Dict[str, Any]:
        """Create Moodle webhook subscription (using external services)"""
        # Moodle doesn't have built-in webhooks, but we can set up external service notifications
        params = {
            "wstoken": self.api_key,
            "wsfunction": "core_webservice_get_site_info",
            "moodlewsrestformat": "json"
        }
        
        async with self._session.get(
            f"{self.base_url}/webservice/rest/server.php",
            params=params
        ) as response:
            if response.status == 200:
                # For Moodle, we'll return configuration info for manual webhook setup
                return {
                    "webhook_url": webhook_url,
                    "events": events,
                    "setup_required": True,
                    "instructions": "Manual webhook configuration required in Moodle admin panel"
                }
            else:
                raise LMSAPIError(f"Failed to configure Moodle webhook: {response.status}")


class LMSIntegrationService:
    """Main service for managing LMS integrations"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseLMSConnector] = {}
        self.webhook_handlers: Dict[str, callable] = {}
    
    def register_connector(self, lms_type: str, connector: BaseLMSConnector):
        """Register an LMS connector"""
        self.connectors[lms_type] = connector
    
    def get_connector(self, lms_type: str) -> BaseLMSConnector:
        """Get LMS connector by type"""
        if lms_type not in self.connectors:
            raise LMSIntegrationError(f"No connector registered for LMS type: {lms_type}")
        return self.connectors[lms_type]
    
    async def sync_assignment(self, lms_type: str, course_id: str, assignment_id: str) -> LMSAssignment:
        """Sync assignment from LMS"""
        connector = self.get_connector(lms_type)
        async with connector:
            assignments = await connector.get_assignments(course_id)
            for assignment in assignments:
                if assignment.id == assignment_id:
                    return assignment
            raise LMSIntegrationError(f"Assignment {assignment_id} not found in course {course_id}")
    
    async def submit_verification_result(
        self, 
        lms_type: str, 
        assignment_id: str, 
        user_id: str, 
        submission: Submission
    ) -> bool:
        """Submit verification result back to LMS as grade"""
        connector = self.get_connector(lms_type)
        
        # Calculate grade based on verification results
        verification_result = submission.verification_result
        if not verification_result:
            return False
        
        # Grade calculation logic
        base_score = 100
        if verification_result.overall_status == "FAIL":
            base_score = 0
        elif verification_result.overall_status == "REVIEW":
            base_score = 50
        
        # Adjust based on authorship and AI detection scores
        authorship_penalty = max(0, (70 - verification_result.authorship_score) * 0.5)
        ai_penalty = max(0, verification_result.ai_probability * 0.3)
        
        final_score = max(0, base_score - authorship_penalty - ai_penalty)
        
        # Create grade comment
        comment_parts = [
            f"Authorship Score: {verification_result.authorship_score:.1f}%",
            f"AI Detection: {verification_result.ai_probability:.1f}%",
            f"Status: {verification_result.overall_status}"
        ]
        
        if verification_result.duplicate_matches:
            comment_parts.append(f"Duplicate matches found: {len(verification_result.duplicate_matches)}")
        
        grade = LMSGrade(
            assignment_id=assignment_id,
            user_id=user_id,
            course_id=submission.course_id if hasattr(submission, 'course_id') else "",
            score=final_score,
            comment="; ".join(comment_parts),
            submitted_at=datetime.now()
        )
        
        async with connector:
            return await connector.submit_grade(assignment_id, user_id, grade)
    
    def register_webhook_handler(self, event_type: str, handler: callable):
        """Register webhook event handler"""
        self.webhook_handlers[event_type] = handler
    
    async def handle_webhook_event(self, lms_type: str, event_data: Dict[str, Any]) -> bool:
        """Handle incoming webhook event"""
        try:
            event = LMSWebhookEvent(**event_data)
            
            if event.event_type in self.webhook_handlers:
                handler = self.webhook_handlers[event.event_type]
                await handler(event)
                return True
            else:
                logger.warning(f"No handler registered for event type: {event.event_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error handling webhook event: {str(e)}")
            return False


# Global service instance
lms_integration_service = LMSIntegrationService()
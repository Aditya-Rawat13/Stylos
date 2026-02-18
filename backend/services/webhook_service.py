"""
Webhook Service

Handles webhook events from LMS platforms for real-time communication.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.config import settings
from core.database import get_db
from models.submission import Submission
from models.user import User
from schemas.lms import LMSWebhookEvent, LMSEventTypes, LMSType
from services.verification_service import verification_service

logger = logging.getLogger(__name__)


class WebhookError(Exception):
    """Base exception for webhook errors"""
    pass


class WebhookSignatureError(WebhookError):
    """Webhook signature verification failed"""
    pass


class WebhookEvent(BaseModel):
    """Internal webhook event representation"""
    id: str
    event_type: str
    lms_type: LMSType
    institution_id: str
    timestamp: datetime
    data: Dict[str, Any]
    processed: bool = False
    retry_count: int = 0
    max_retries: int = 3


class WebhookSubscription(BaseModel):
    """Webhook subscription configuration"""
    id: str
    lms_type: LMSType
    institution_id: str
    webhook_url: str
    events: List[str]
    secret: Optional[str] = None
    active: bool = True
    created_at: datetime
    last_delivery: Optional[datetime] = None


class WebhookDelivery(BaseModel):
    """Webhook delivery attempt"""
    id: str
    subscription_id: str
    event_id: str
    status: str  # 'pending', 'success', 'failed'
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: Optional[datetime] = None
    retry_count: int = 0


class WebhookService:
    """Service for managing webhook events and subscriptions"""
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.signature_validators: Dict[LMSType, Callable] = {
            LMSType.CANVAS: self._validate_canvas_signature,
            LMSType.BLACKBOARD: self._validate_blackboard_signature,
            LMSType.MOODLE: self._validate_moodle_signature
        }
        self.event_queue: List[WebhookEvent] = []
        self.processing_active = False
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    def validate_webhook_signature(
        self, 
        lms_type: LMSType, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """Validate webhook signature based on LMS type"""
        validator = self.signature_validators.get(lms_type)
        if not validator:
            logger.warning(f"No signature validator for LMS type: {lms_type}")
            return True  # Allow if no validator (not recommended for production)
        
        return validator(payload, signature, secret)
    
    def _validate_canvas_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Validate Canvas webhook signature"""
        try:
            # Canvas uses HMAC-SHA256
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Canvas sends signature as 'sha256=<hash>'
            if signature.startswith('sha256='):
                signature = signature[7:]
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Canvas signature validation error: {str(e)}")
            return False
    
    def _validate_blackboard_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Validate Blackboard webhook signature"""
        try:
            # Blackboard uses HMAC-SHA256
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Blackboard signature validation error: {str(e)}")
            return False
    
    def _validate_moodle_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Validate Moodle webhook signature"""
        try:
            # Moodle typically uses HMAC-SHA1 or SHA256
            expected_signature = hmac.new(
                secret.encode(),
                payload,
                hashlib.sha1
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Moodle signature validation error: {str(e)}")
            return False
    
    async def process_webhook_event(
        self, 
        lms_type: LMSType, 
        event_data: Dict[str, Any],
        signature: Optional[str] = None,
        secret: Optional[str] = None
    ) -> bool:
        """Process incoming webhook event"""
        try:
            # Create webhook event
            event = WebhookEvent(
                id=event_data.get('id', f"evt_{datetime.now().timestamp()}"),
                event_type=event_data.get('event_type', 'unknown'),
                lms_type=lms_type,
                institution_id=event_data.get('institution_id', 'default'),
                timestamp=datetime.now(),
                data=event_data
            )
            
            # Add to processing queue
            self.event_queue.append(event)
            
            # Start processing if not already active
            if not self.processing_active:
                asyncio.create_task(self._process_event_queue())
            
            return True
        
        except Exception as e:
            logger.error(f"Error processing webhook event: {str(e)}")
            return False
    
    async def _process_event_queue(self):
        """Process events in the queue"""
        self.processing_active = True
        
        try:
            while self.event_queue:
                event = self.event_queue.pop(0)
                await self._handle_single_event(event)
        
        except Exception as e:
            logger.error(f"Error processing event queue: {str(e)}")
        
        finally:
            self.processing_active = False
    
    async def _handle_single_event(self, event: WebhookEvent):
        """Handle a single webhook event"""
        try:
            logger.info(f"Processing webhook event: {event.event_type} from {event.lms_type}")
            
            # Get handlers for this event type
            handlers = self.event_handlers.get(event.event_type, [])
            
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type}")
                return
            
            # Execute all handlers
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Handler error for event {event.event_type}: {str(e)}")
                    
                    # Retry logic
                    if event.retry_count < event.max_retries:
                        event.retry_count += 1
                        # Add back to queue for retry (with delay)
                        await asyncio.sleep(2 ** event.retry_count)  # Exponential backoff
                        self.event_queue.append(event)
                    else:
                        logger.error(f"Max retries exceeded for event {event.id}")
            
            event.processed = True
            logger.info(f"Successfully processed event: {event.id}")
        
        except Exception as e:
            logger.error(f"Error handling event {event.id}: {str(e)}")
    
    async def send_webhook(
        self, 
        webhook_url: str, 
        event_data: Dict[str, Any], 
        secret: Optional[str] = None
    ) -> bool:
        """Send webhook to external URL"""
        try:
            payload = json.dumps(event_data).encode()
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Project-Stylos-Webhook/1.0'
            }
            
            # Add signature if secret provided
            if secret:
                signature = hmac.new(
                    secret.encode(),
                    payload,
                    hashlib.sha256
                ).hexdigest()
                headers['X-Stylos-Signature'] = f'sha256={signature}'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    data=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook delivered successfully to {webhook_url}")
                        return True
                    else:
                        logger.error(f"Webhook delivery failed: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook_url}: {str(e)}")
            return False


# Event handler implementations

async def handle_assignment_created(event: WebhookEvent):
    """Handle assignment created event"""
    try:
        assignment_data = event.data
        logger.info(f"New assignment created: {assignment_data.get('name')}")
        
        # Store assignment in database if needed
        # This could trigger automatic setup for the assignment
        
    except Exception as e:
        logger.error(f"Error handling assignment created: {str(e)}")


async def handle_submission_created(event: WebhookEvent):
    """Handle submission created event"""
    try:
        submission_data = event.data
        logger.info(f"New submission created: {submission_data.get('id')}")
        
        # Trigger automatic verification
        assignment_id = submission_data.get('assignment_id')
        user_id = submission_data.get('user_id')
        
        if assignment_id and user_id:
            # Check if this assignment is configured for automatic verification
            # If so, trigger the verification process
            await trigger_automatic_verification(assignment_id, user_id, submission_data)
        
    except Exception as e:
        logger.error(f"Error handling submission created: {str(e)}")


async def handle_grade_changed(event: WebhookEvent):
    """Handle grade changed event"""
    try:
        grade_data = event.data
        logger.info(f"Grade changed for assignment: {grade_data.get('assignment_id')}")
        
        # Sync grade changes if needed
        # This could update local records or trigger notifications
        
    except Exception as e:
        logger.error(f"Error handling grade changed: {str(e)}")


async def handle_user_enrolled(event: WebhookEvent):
    """Handle user enrollment event"""
    try:
        enrollment_data = event.data
        logger.info(f"User enrolled: {enrollment_data.get('user_id')} in course {enrollment_data.get('course_id')}")
        
        # Create or update user profile
        # Set up writing profile if needed
        
    except Exception as e:
        logger.error(f"Error handling user enrolled: {str(e)}")


async def trigger_automatic_verification(assignment_id: str, user_id: str, submission_data: Dict[str, Any]):
    """Trigger automatic verification for new submission"""
    try:
        # Get database session
        db = next(get_db())
        
        # Check if user exists and has writing profile
        user = db.query(User).filter(User.lms_user_id == user_id).first()
        if not user:
            logger.warning(f"User not found for LMS user ID: {user_id}")
            return
        
        # Check if submission already exists
        existing_submission = db.query(Submission).filter(
            Submission.lms_assignment_id == assignment_id,
            Submission.student_id == user.id
        ).first()
        
        if existing_submission:
            logger.info(f"Submission already exists for assignment {assignment_id}, user {user_id}")
            return
        
        # Create submission record
        submission = Submission(
            student_id=user.id,
            title=submission_data.get('title', f'Assignment {assignment_id}'),
            content=submission_data.get('body', ''),
            lms_assignment_id=assignment_id,
            lms_submission_id=submission_data.get('id'),
            status='PENDING'
        )
        
        db.add(submission)
        db.commit()
        
        # Trigger verification
        await verification_service.verify_submission(submission.id, db)
        
        logger.info(f"Automatic verification triggered for submission {submission.id}")
        
    except Exception as e:
        logger.error(f"Error triggering automatic verification: {str(e)}")


# Global service instance
webhook_service = WebhookService()

# Register default event handlers
webhook_service.register_event_handler(LMSEventTypes.ASSIGNMENT_CREATED, handle_assignment_created)
webhook_service.register_event_handler(LMSEventTypes.SUBMISSION_CREATED, handle_submission_created)
webhook_service.register_event_handler(LMSEventTypes.GRADE_CHANGED, handle_grade_changed)
webhook_service.register_event_handler(LMSEventTypes.USER_ENROLLED, handle_user_enrolled)
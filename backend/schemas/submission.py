"""
Submission schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from models.submission import SubmissionStatus


class SubmissionResponse(BaseModel):
    """Complete submission response with all related data."""
    id: int
    user_id: int
    filename: str
    title: Optional[str]
    content: str
    word_count: int
    file_hash: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    # Verification results
    verification_result: Optional[Dict[str, Any]] = None
    authorship_score: Optional[float] = None
    ai_probability: Optional[float] = None
    similarity_score: Optional[float] = None
    is_authentic: Optional[bool] = None
    has_duplicates: Optional[bool] = None
    is_ai_generated: Optional[bool] = None
    
    # Blockchain information
    blockchain_record: Optional[Dict[str, Any]] = None
    transaction_hash: Optional[str] = None
    ipfs_hash: Optional[str] = None
    blockchain_status: Optional[str] = None
    
    # Processing information
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm_with_relations(cls, submission, verification=None, blockchain=None):
        """Create response from ORM objects with all relations."""
        data = {
            'id': submission.id,
            'user_id': submission.user_id,
            'filename': submission.filename,
            'title': submission.title,
            'content': submission.content[:500] if submission.content else "",  # Truncate for list view
            'word_count': submission.word_count,
            'file_hash': submission.file_hash,
            'status': submission.status.value if hasattr(submission.status, 'value') else str(submission.status),
            'created_at': submission.created_at,
            'updated_at': submission.updated_at,
            'processing_started_at': submission.processing_started_at,
            'processing_completed_at': submission.processing_completed_at,
        }
        
        # Add verification results if available
        if verification:
            data.update({
                'verification_result': {
                    'status': verification.status.value if hasattr(verification.status, 'value') else str(verification.status),
                    'overall_risk_score': verification.overall_risk_score,
                    'is_flagged': verification.is_flagged,
                    'processing_time': verification.processing_time_seconds,
                    'model_versions': verification.model_versions
                },
                'authorship_score': verification.authorship_score,
                'ai_probability': verification.ai_probability,
                'similarity_score': verification.similarity_score,
                'is_authentic': verification.is_authentic,
                'has_duplicates': verification.has_duplicates,
                'is_ai_generated': verification.is_ai_generated,
            })
        
        # Add blockchain information if available
        if blockchain:
            data.update({
                'blockchain_record': {
                    'status': blockchain.status.value if hasattr(blockchain.status, 'value') else str(blockchain.status),
                    'network': blockchain.network_name,
                    'explorer_url': blockchain.explorer_url,
                    'ipfs_url': blockchain.ipfs_url,
                    'confirmed_at': blockchain.confirmed_at
                },
                'transaction_hash': blockchain.transaction_hash,
                'ipfs_hash': blockchain.ipfs_hash,
                'blockchain_status': blockchain.status.value if hasattr(blockchain.status, 'value') else str(blockchain.status),
            })
        
        return cls(**data)


class SubmissionListResponse(BaseModel):
    """Response for submission list with pagination."""
    submissions: List[SubmissionResponse]
    total: int
    page: int
    total_pages: int
    has_next: bool
    has_prev: bool


class SubmissionDetailResponse(SubmissionResponse):
    """Detailed submission response with full content."""
    content: str  # Full content, not truncated
    stylometric_analysis: Optional[Dict[str, Any]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None
    duplicate_submissions: Optional[List[int]] = None
    admin_notes: Optional[str] = None
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None


class SubmissionUploadRequest(BaseModel):
    """Request for uploading a submission."""
    title: Optional[str] = None
    assignment_id: Optional[str] = None
    assignment_title: Optional[str] = None
    course_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SubmissionUploadResponse(BaseModel):
    """Response after successful upload."""
    submission_id: str
    message: str
    filename: str
    file_hash: str
    word_count: int
    status: str
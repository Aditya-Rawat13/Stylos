"""
API endpoints for authorship verification and AI detection.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
import numpy as np

from services.verification_service import (
    verification_service, 
    VerificationRequest,
    ComprehensiveVerificationResult
)
from services.authorship_models import AuthorshipResult, AIDetectionResult
from core.auth import get_current_user
from models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class VerificationRequestModel(BaseModel):
    """Request model for text verification."""
    text: str = Field(..., min_length=100, description="Text to verify (minimum 100 characters)")
    student_id: str = Field(..., description="Student identifier")
    submission_id: str = Field(..., description="Unique submission identifier")
    reference_texts: Optional[List[str]] = Field(None, description="Optional reference texts from the same author")

class AuthorshipResultModel(BaseModel):
    """Response model for authorship verification result."""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0 and 1")
    confidence_interval: tuple[float, float] = Field(..., description="Confidence interval for the similarity score")
    is_authentic: bool = Field(..., description="Whether the text is considered authentic")
    uncertainty: float = Field(..., ge=0.0, description="Uncertainty measure")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")

class AIDetectionResultModel(BaseModel):
    """Response model for AI detection result."""
    ai_probability: float = Field(..., ge=0.0, le=1.0, description="Probability that content is AI-generated")
    human_probability: float = Field(..., ge=0.0, le=1.0, description="Probability that content is human-written")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the detection")
    detection_method: str = Field(..., description="Method used for detection")
    explanation: Dict[str, Any] = Field(..., description="Explanation of the detection result")

class VerificationResultModel(BaseModel):
    """Response model for comprehensive verification result."""
    submission_id: str
    student_id: str
    timestamp: str
    authorship_result: AuthorshipResultModel
    ai_detection_result: AIDetectionResultModel
    duplicate_matches: List[Dict[str, Any]]
    stylometric_features: Dict[str, float]
    overall_status: str = Field(..., pattern="^(PASS|FAIL|REVIEW|ERROR)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_factors: List[str]

class TrainingDataModel(BaseModel):
    """Model for training data submission."""
    authorship_samples: Optional[Dict[str, Any]] = None
    ai_detection_samples: Optional[Dict[str, Any]] = None

class ThresholdUpdateModel(BaseModel):
    """Model for updating verification thresholds."""
    authorship_min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_detection_max_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    duplicate_max_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

@router.post("/verify", response_model=VerificationResultModel)
async def verify_submission(
    request: VerificationRequestModel,
    current_user: User = Depends(get_current_user)
):
    """
    Perform comprehensive verification of a text submission.
    
    This endpoint performs:
    - Authorship verification using Siamese networks
    - AI-generated content detection
    - Duplicate/plagiarism detection
    - Stylometric analysis
    """
    try:
        logger.info(f"Verification request from user {current_user.id} for submission {request.submission_id}")
        
        # Create verification request
        verification_request = VerificationRequest(
            text=request.text,
            student_id=request.student_id,
            submission_id=request.submission_id,
            reference_texts=request.reference_texts
        )
        
        # Perform verification
        result = await verification_service.verify_submission(verification_request)
        
        # Convert to response model
        response_data = result.to_dict()
        
        return VerificationResultModel(**response_data)
        
    except Exception as e:
        logger.error(f"Error in verification endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@router.post("/verify/file", response_model=VerificationResultModel)
async def verify_file_submission(
    file: UploadFile = File(...),
    student_id: str = Form(...),
    submission_id: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """
    Verify a text submission from an uploaded file.
    
    Supports PDF, DOCX, and TXT files.
    """
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, or TXT files.")
        
        # Read and process file
        content = await file.read()
        
        # Extract text from file (this would use the text processor service)
        from services.text_processor import TextProcessor
        text_processor = TextProcessor()
        extracted_text = await text_processor.extract_text_from_file(content, file.content_type)
        
        if len(extracted_text) < 100:
            raise HTTPException(status_code=400, detail="Extracted text is too short (minimum 100 characters required)")
        
        # Create verification request
        verification_request = VerificationRequest(
            text=extracted_text,
            student_id=student_id,
            submission_id=submission_id
        )
        
        # Perform verification
        result = await verification_service.verify_submission(verification_request)
        
        # Convert to response model
        response_data = result.to_dict()
        
        return VerificationResultModel(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file verification endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"File verification failed: {str(e)}")

@router.get("/authorship/{student_id}/{submission_id}", response_model=AuthorshipResultModel)
async def get_authorship_result(
    student_id: str,
    submission_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed authorship verification result for a specific submission.
    """
    try:
        # This would typically query the database for stored results
        # For now, return a placeholder response
        logger.info(f"Authorship result request for submission {submission_id}")
        
        # Placeholder result
        result = AuthorshipResultModel(
            similarity_score=0.85,
            confidence_interval=(0.75, 0.95),
            is_authentic=True,
            uncertainty=0.1,
            feature_importance={
                "stylometric_similarity": 0.8,
                "semantic_similarity": 0.9,
                "writing_pattern_match": 0.85
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting authorship result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get authorship result: {str(e)}")

@router.get("/ai-detection/{student_id}/{submission_id}", response_model=AIDetectionResultModel)
async def get_ai_detection_result(
    student_id: str,
    submission_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed AI detection result for a specific submission.
    """
    try:
        # This would typically query the database for stored results
        logger.info(f"AI detection result request for submission {submission_id}")
        
        # Placeholder result
        result = AIDetectionResultModel(
            ai_probability=0.15,
            human_probability=0.85,
            confidence=0.9,
            detection_method="ensemble",
            explanation={
                "stylometric_indicators": "Human-like writing patterns",
                "transformer_analysis": "Low AI probability",
                "ensemble_decision": "Human-written content"
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting AI detection result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI detection result: {str(e)}")

@router.post("/train")
async def train_verification_models(
    training_data: TrainingDataModel,
    current_user: User = Depends(get_current_user)
):
    """
    Train verification models with new data.
    
    Requires admin privileges.
    """
    try:
        # Check if user has admin privileges
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        logger.info(f"Model training request from admin {current_user.id}")
        
        # Convert to dictionary format
        training_dict = training_data.dict(exclude_none=True)
        
        # Start training
        await verification_service.train_models(training_dict)
        
        return {"message": "Model training completed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.put("/thresholds")
async def update_verification_thresholds(
    thresholds: ThresholdUpdateModel,
    current_user: User = Depends(get_current_user)
):
    """
    Update verification thresholds.
    
    Requires admin privileges.
    """
    try:
        # Check if user has admin privileges
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        logger.info(f"Threshold update request from admin {current_user.id}")
        
        # Update thresholds
        threshold_dict = thresholds.dict(exclude_none=True)
        verification_service.update_thresholds(threshold_dict)
        
        return {"message": "Verification thresholds updated successfully", "updated_thresholds": threshold_dict}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")

@router.get("/statistics")
async def get_verification_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get verification statistics and performance metrics.
    
    Requires admin privileges.
    """
    try:
        # Check if user has admin privileges
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        logger.info(f"Statistics request from admin {current_user.id}")
        
        # Get statistics
        stats = await verification_service.get_verification_statistics()
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting verification statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/initialize")
async def initialize_verification_service(
    current_user: User = Depends(get_current_user)
):
    """
    Initialize the verification service and all models.
    
    This endpoint can be used to manually initialize or reinitialize the service.
    """
    try:
        logger.info(f"Verification service initialization request from user {current_user.id}")
        
        # Initialize the service
        await verification_service.initialize()
        
        return {"message": "Verification service initialized successfully"}
        
    except Exception as e:
        logger.error(f"Error initializing verification service: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.get("/health")
async def verification_health_check():
    """
    Health check endpoint for verification service.
    """
    try:
        # Check if service is initialized
        is_initialized = verification_service.initialized
        
        return {
            "status": "healthy" if is_initialized else "not_initialized",
            "initialized": is_initialized,
            "timestamp": str(np.datetime64('now'))
        }
        
    except Exception as e:
        logger.error(f"Error in verification health check: {e}")
        return {
            "status": "error",
            "initialized": False,
            "error": str(e)
        }
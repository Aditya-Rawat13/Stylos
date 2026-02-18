"""
FIXED Submission management endpoints - WORKING VERSION
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime
import hashlib
import os
from pathlib import Path
import logging
import asyncio

from core.database import get_db, AsyncSessionLocal
from core.auth import get_current_user
from models.submission import Submission, SubmissionStatus
from models.verification import VerificationResult, VerificationStatus
from models.blockchain import BlockchainRecord
from models.user import User
from schemas.submission import SubmissionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Global progress tracking
submission_progress = {}


@router.post("/upload")
async def upload_submission(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    course_id: Optional[str] = None,
    assignment_title: Optional[str] = None,
    assignment_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a new submission with REAL AI detection and verification."""
    
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate file
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 10MB limit")
        
        # Calculate hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Extract text
        from services.file_processor import extract_text_from_file
        text_content = await extract_text_from_file(file_content, file.filename)
        
        if not text_content or len(text_content.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="File content too short (minimum 50 characters required)"
            )
        
        word_count = len(text_content.split())
        
        # Save file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{file_hash}{file_ext}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Create submission
        submission_title = title or os.path.splitext(file.filename)[0]
        
        submission = Submission(
            user_id=current_user.id,
            filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            file_hash=file_hash,
            title=submission_title,
            content=text_content[:10000],
            word_count=word_count,
            course_id=course_id,
            assignment_title=assignment_title,
            assignment_id=assignment_id,
            status=SubmissionStatus.PROCESSING,
            processing_started_at=datetime.utcnow(),
            submission_metadata={
                "uploaded_at": datetime.utcnow().isoformat(),
                "original_filename": file.filename,
                "file_extension": file_ext
            }
        )
        
        db.add(submission)
        await db.commit()
        await db.refresh(submission)
        
        logger.info(f"✓ Submission {submission.id} created for user {current_user.id}")
        
        # Initialize progress tracking
        submission_progress[submission.id] = {
            "stage": "UPLOADING",
            "progress": 10,
            "message": "File uploaded, starting verification..."
        }
        
        # Start background verification
        asyncio.create_task(
            process_submission_with_real_models(
                submission.id,
                text_content,
                current_user.id
            )
        )
        
        return {
            "submissionId": str(submission.id),
            "message": "Upload successful. Real AI detection starting...",
            "filename": file.filename,
            "fileHash": file_hash,
            "wordCount": word_count,
            "status": "PROCESSING"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_submission_with_real_models(
    submission_id: int,
    text_content: str,
    user_id: int
):
    """
    FIXED: Process submission with REAL models, no mocks.
    Updates progress in real-time.
    """
    async with AsyncSessionLocal() as db:
        try:
            logger.info(f"🚀 Starting REAL verification for submission {submission_id}")
            
            # Update progress
            submission_progress[submission_id] = {
                "stage": "PROCESSING",
                "progress": 20,
                "message": "Initializing AI detection models..."
            }
            
            # Get submission
            result = await db.execute(
                select(Submission).filter(Submission.id == submission_id)
            )
            submission = result.scalar_one_or_none()
            
            if not submission:
                logger.error(f"Submission {submission_id} not found")
                return
            
            # STEP 1: Initialize ALL models (NO MOCKS)
            logger.info("Step 1/6: Loading AI models...")
            submission_progress[submission_id]["progress"] = 25
            
            from services.ai_detection_enhanced import EnhancedAIDetectionClassifier
            from services.authorship_models import AuthorshipVerificationService
            from services.stylometric_analyzer import stylometric_analyzer
            from services.embedding_service import embedding_service
            from services.text_processor import TextProcessor
            
            # Check if models are available
            try:
                import torch
                import transformers
                models_available = True
                logger.info("✓ PyTorch and Transformers available")
            except ImportError as e:
                models_available = False
                logger.error(f"✗ ML libraries not available: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="AI models not installed. Install PyTorch and Transformers."
                )
            
            # Initialize services
            ai_detector = EnhancedAIDetectionClassifier()
            await ai_detector.initialize_models()
            
            authorship_service = AuthorshipVerificationService()
            await authorship_service.initialize()
            
            await embedding_service.initialize_models()
            
            logger.info("✓ All models initialized")
            
            # STEP 2: Process text
            logger.info("Step 2/6: Processing text...")
            submission_progress[submission_id]["progress"] = 35
            
            text_processor = TextProcessor()
            processed_text = await text_processor.process_text(text_content)
            
            # STEP 3: Extract stylometric features
            logger.info("Step 3/6: Extracting stylometric features...")
            submission_progress[submission_id]["progress"] = 45
            
            stylometric_features = stylometric_analyzer.extract_features(processed_text)
            logger.info(f"✓ Extracted {len(stylometric_features)} features")
            
            # STEP 4: Run AI detection
            logger.info("Step 4/6: Running AI detection...")
            submission_progress[submission_id]["progress"] = 60
            
            ai_probability, ai_confidence, ai_explanation = await ai_detector.detect_ai_content_enhanced(processed_text)
            
            logger.info(f"✓ AI Detection: {ai_probability:.1%} probability, {ai_confidence:.1%} confidence")
            logger.info(f"  Indicators: {ai_explanation.primary_indicators}")
            
            # STEP 5: Generate embeddings and verify authorship
            logger.info("Step 5/6: Verifying authorship...")
            submission_progress[submission_id]["progress"] = 75
            
            text_embedding = await embedding_service.generate_embedding(processed_text)
            
            # Get user's reference embeddings (if any)
            from models.user import WritingProfile
            result = await db.execute(
                select(WritingProfile).filter(WritingProfile.user_id == user_id)
            )
            writing_profile = result.scalar_one_or_none()
            
            reference_embeddings = []
            if writing_profile and writing_profile.is_initialized:
                # Load reference embeddings from profile
                if writing_profile.sample_essays:
                    for essay in writing_profile.sample_essays:
                        ref_embedding = await embedding_service.generate_embedding(essay.get('content', ''))
                        reference_embeddings.append(ref_embedding)
            
            # Verify authorship
            authorship_result = await authorship_service.verify_authorship(
                text_embedding,
                reference_embeddings
            )
            
            logger.info(f"✓ Authorship: {authorship_result.similarity_score:.1%} similarity")
            
            # STEP 6: Check for duplicates
            logger.info("Step 6/6: Checking for duplicates...")
            submission_progress[submission_id]["progress"] = 85
            
            from services.duplicate_detection_service import DuplicateDetectionService
            duplicate_service = DuplicateDetectionService()
            await duplicate_service.initialize()
            
            duplicate_matches = await duplicate_service.find_duplicates(
                text_embedding,
                processed_text,
                exclude_student_id=str(user_id),
                exclude_submission_id=str(submission_id)
            )
            
            logger.info(f"✓ Found {len(duplicate_matches)} potential duplicates")
            
            # STEP 7: Calculate overall result
            submission_progress[submission_id]["progress"] = 90
            
            # Determine overall status
            risk_factors = []
            
            if ai_probability > 0.7:
                risk_factors.append(f"High AI probability ({ai_probability:.1%})")
            
            if not authorship_result.is_authentic and len(reference_embeddings) > 0:
                risk_factors.append("Authorship mismatch with profile")
            
            if len(duplicate_matches) > 0:
                risk_factors.append(f"{len(duplicate_matches)} similar submissions found")
            
            # Overall status
            if len(risk_factors) == 0:
                overall_status = "PASS"
                submission_status = SubmissionStatus.VERIFIED
            elif len(risk_factors) >= 2 or ai_probability > 0.8:
                overall_status = "FAIL"
                submission_status = SubmissionStatus.REJECTED
            else:
                overall_status = "REVIEW"
                submission_status = SubmissionStatus.FLAGGED
            
            # STEP 8: Save results
            logger.info("Saving verification results...")
            
            verification = VerificationResult(
                submission_id=submission_id,
                status=VerificationStatus.COMPLETED,
                authorship_score=authorship_result.similarity_score,
                authorship_confidence=1.0 - authorship_result.uncertainty,
                is_authentic=authorship_result.is_authentic,
                ai_probability=ai_probability,
                is_ai_generated=ai_probability > 0.7,
                ai_detection_model="enhanced_multi_model_v2",
                has_duplicates=len(duplicate_matches) > 0,
                duplicate_submissions=duplicate_matches,
                stylometric_analysis={
                    **stylometric_features,
                    'ai_explanation': {
                        'indicators': ai_explanation.primary_indicators,
                        'risk': ai_explanation.risk_assessment,
                        'summary': ai_explanation.human_readable_summary
                    }
                },
                processing_time_seconds=(datetime.utcnow() - submission.processing_started_at).total_seconds(),
                model_versions={
                    'ai_detection': 'enhanced_v2.0',
                    'authorship': 'siamese_v2.1',
                    'stylometric': 'comprehensive_v1.0'
                },
                completed_at=datetime.utcnow()
            )
            
            db.add(verification)
            
            # Update submission
            submission.status = submission_status
            submission.ai_detection_score = ai_probability
            submission.confidence_score = ai_confidence
            submission.stylometric_features = stylometric_features
            submission.processing_completed_at = datetime.utcnow()
            submission.verification_summary = {
                'overall_status': overall_status,
                'risk_factors': risk_factors,
                'ai_probability': ai_probability,
                'authorship_score': authorship_result.similarity_score
            }
            
            await db.commit()
            
            # Update progress
            submission_progress[submission_id] = {
                "stage": "COMPLETED",
                "progress": 100,
                "message": f"Verification complete: {overall_status}"
            }
            
            logger.info(f"✓ Verification COMPLETE for submission {submission_id}")
            logger.info(f"  Status: {overall_status}")
            logger.info(f"  AI: {ai_probability:.1%}")
            logger.info(f"  Authorship: {authorship_result.similarity_score:.1%}")
            logger.info(f"  Risk Factors: {len(risk_factors)}")
            
        except Exception as e:
            logger.error(f"✗ Verification FAILED for submission {submission_id}: {e}", exc_info=True)
            
            # Update submission to error state
            try:
                result = await db.execute(
                    select(Submission).filter(Submission.id == submission_id)
                )
                submission = result.scalar_one_or_none()
                
                if submission:
                    submission.status = SubmissionStatus.ERROR
                    submission.processing_completed_at = datetime.utcnow()
                    submission.verification_summary = {
                        'error': str(e),
                        'overall_status': 'ERROR'
                    }
                    await db.commit()
                
                submission_progress[submission_id] = {
                    "stage": "ERROR",
                    "progress": 0,
                    "message": f"Verification failed: {str(e)}"
                }
                
            except Exception as db_error:
                logger.error(f"Failed to update error status: {db_error}")


@router.get("/{submission_id}/progress")
async def get_submission_progress(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get REAL processing progress for a submission."""
    
    # Check if we have real-time progress
    if submission_id in submission_progress:
        return submission_progress[submission_id]
    
    # Otherwise check database status
    result = await db.execute(
        select(Submission).filter(
            Submission.id == submission_id,
            Submission.user_id == current_user.id
        )
    )
    submission = result.scalar_one_or_none()
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Map status to progress
    status_map = {
        SubmissionStatus.UPLOADED: {"stage": "UPLOADING", "progress": 10},
        SubmissionStatus.PROCESSING: {"stage": "PROCESSING", "progress": 50},
        SubmissionStatus.VERIFIED: {"stage": "COMPLETED", "progress": 100},
        SubmissionStatus.FLAGGED: {"stage": "COMPLETED", "progress": 100},
        SubmissionStatus.REJECTED: {"stage": "COMPLETED", "progress": 100},
        SubmissionStatus.ERROR: {"stage": "ERROR", "progress": 0},
    }
    
    progress_info = status_map.get(submission.status, {"stage": "UNKNOWN", "progress": 0})
    progress_info["message"] = f"Status: {submission.status.value if hasattr(submission.status, 'value') else submission.status}"
    
    return progress_info


@router.get("")
async def get_submissions(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's submissions with pagination."""
    
    query = select(Submission).filter(Submission.user_id == current_user.id)
    
    if status:
        query = query.filter(Submission.status == status)
    
    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(Submission).filter(Submission.user_id == current_user.id)
    )
    total = count_result.scalar()
    
    # Get submissions
    offset = (page - 1) * limit
    query = query.order_by(Submission.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    submissions = result.scalars().all()
    
    # Load related data
    submission_list = []
    for sub in submissions:
        # Get verification
        ver_result = await db.execute(
            select(VerificationResult).filter(VerificationResult.submission_id == sub.id)
        )
        verification = ver_result.scalar_one_or_none()
        
        # Get blockchain
        bc_result = await db.execute(
            select(BlockchainRecord).filter(BlockchainRecord.submission_id == sub.id)
        )
        blockchain = bc_result.scalar_one_or_none()
        
        submission_list.append(SubmissionResponse.from_orm_with_relations(sub, verification, blockchain).dict())
    
    return {
        "submissions": submission_list,
        "total": total,
        "page": page,
        "totalPages": (total + limit - 1) // limit
    }


@router.get("/{submission_id}")
async def get_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific submission with REAL verification results."""
    
    result = await db.execute(
        select(Submission).filter(
            Submission.id == submission_id,
            Submission.user_id == current_user.id
        )
    )
    submission = result.scalar_one_or_none()
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Get verification result
    ver_result = await db.execute(
        select(VerificationResult).filter(VerificationResult.submission_id == submission_id)
    )
    verification = ver_result.scalar_one_or_none()
    
    return {
        "id": str(submission.id),
        "title": submission.title or submission.filename,
        "content": submission.content,
        "contentHash": submission.file_hash,
        "status": submission.status.value if hasattr(submission.status, 'value') else str(submission.status),
        "submittedAt": submission.created_at.isoformat() if submission.created_at else None,
        "updatedAt": submission.updated_at.isoformat() if submission.updated_at else None,
        "verificationResult": {
            "authorshipScore": verification.authorship_score if verification else 0.0,
            "aiProbability": verification.ai_probability if verification else 0.0,
            "duplicateMatches": verification.duplicate_submissions if verification else [],
            "overallStatus": submission.verification_summary.get('overall_status', 'PENDING') if submission.verification_summary else 'PENDING',
            "confidence": verification.authorship_confidence if verification else 0.0,
            "riskFactors": submission.verification_summary.get('risk_factors', []) if submission.verification_summary else []
        } if verification else None
    }

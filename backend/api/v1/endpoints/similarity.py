"""
API endpoints for similarity analysis and duplicate detection.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user
from core.database import get_db
from models.user import User
from services.duplicate_detection_service import (
    duplicate_detection_service, 
    DetectionLevel,
    PlagiarismAlert,
    DuplicateAnalysisReport
)
from services.similarity_service import similarity_service
from services.embedding_service import embedding_service

router = APIRouter()

# Request/Response Models
class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="First text to compare")
    text2: str = Field(..., description="Second text to compare")
    use_semantic: bool = Field(True, description="Use semantic similarity")
    use_lexical: bool = Field(True, description="Use lexical similarity")

class SimilarityResponse(BaseModel):
    similarity_score: float
    is_duplicate: bool
    detection_method: str
    metadata: Dict[str, Any] = {}

class DuplicateDetectionRequest(BaseModel):
    submission_id: str
    submission_text: str
    student_id: str
    existing_submissions: Dict[str, Dict[str, str]]
    exclude_same_student: bool = True

class BatchAnalysisRequest(BaseModel):
    submissions: Dict[str, Dict[str, str]]
    batch_size: int = Field(10, ge=1, le=50)

class DetectionLevelRequest(BaseModel):
    level: str = Field(..., description="Detection level: strict, moderate, or sensitive")

class SearchSimilarRequest(BaseModel):
    query_text: str
    k: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.7, ge=0.0, le=1.0)

# API Endpoints
@router.post("/compare", response_model=SimilarityResponse)
async def compare_texts(
    request: SimilarityRequest,
    current_user: User = Depends(get_current_user)
):
    """Compare similarity between two texts."""
    try:
        if request.use_semantic:
            result = await similarity_service.calculate_semantic_similarity(
                text1=request.text1,
                text2=request.text2
            )
        else:
            result = similarity_service.calculate_lexical_similarity(
                text1=request.text1,
                text2=request.text2
            )
        
        return SimilarityResponse(
            similarity_score=result.similarity_score,
            is_duplicate=result.is_duplicate,
            detection_method=result.detection_method,
            metadata=result.metadata or {}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing texts: {str(e)}")

@router.post("/analyze-submission", response_model=Dict[str, Any])
async def analyze_submission(
    request: DuplicateDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze a submission for duplicates."""
    try:
        report = await duplicate_detection_service.analyze_submission(
            submission_id=request.submission_id,
            submission_text=request.submission_text,
            student_id=request.student_id,
            existing_submissions=request.existing_submissions,
            exclude_same_student=request.exclude_same_student
        )
        
        # Convert to dict for JSON serialization
        return {
            "query_submission_id": report.query_submission_id,
            "total_comparisons": report.total_comparisons,
            "duplicates_found": report.duplicates_found,
            "potential_duplicates": report.potential_duplicates,
            "highest_similarity": report.highest_similarity,
            "processing_time": report.processing_time,
            "alerts": [
                {
                    "submission_id": alert.submission_id,
                    "student_id": alert.student_id,
                    "matched_submission_id": alert.matched_submission_id,
                    "matched_student_id": alert.matched_student_id,
                    "similarity_score": alert.similarity_score,
                    "detection_method": alert.detection_method,
                    "alert_level": alert.alert_level,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata or {}
                }
                for alert in report.alerts
            ],
            "detailed_results": [
                {
                    "text1_id": result.text1_id,
                    "text2_id": result.text2_id,
                    "similarity_score": result.similarity_score,
                    "is_duplicate": result.is_duplicate,
                    "detection_method": result.detection_method,
                    "metadata": result.metadata or {}
                }
                for result in report.detailed_results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing submission: {str(e)}")

@router.post("/batch-analyze", response_model=Dict[str, Any])
async def batch_analyze_submissions(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Analyze multiple submissions in batch."""
    try:
        results = await duplicate_detection_service.batch_analyze_submissions(
            submissions=request.submissions,
            batch_size=request.batch_size
        )
        
        # Convert results to serializable format
        serialized_results = {}
        for sub_id, report in results.items():
            serialized_results[sub_id] = {
                "query_submission_id": report.query_submission_id,
                "total_comparisons": report.total_comparisons,
                "duplicates_found": report.duplicates_found,
                "potential_duplicates": report.potential_duplicates,
                "highest_similarity": report.highest_similarity,
                "processing_time": report.processing_time,
                "alerts_count": len(report.alerts)
            }
        
        return {
            "total_submissions": len(request.submissions),
            "processed_submissions": len(results),
            "results": serialized_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch analysis: {str(e)}")

@router.post("/cross-student-analysis", response_model=Dict[str, Any])
async def cross_student_analysis(
    request: BatchAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Perform cross-student plagiarism analysis."""
    try:
        student_alerts = await duplicate_detection_service.cross_student_analysis(
            submissions=request.submissions
        )
        
        # Convert to serializable format
        serialized_alerts = {}
        for student_id, alerts in student_alerts.items():
            serialized_alerts[student_id] = [
                {
                    "submission_id": alert.submission_id,
                    "matched_submission_id": alert.matched_submission_id,
                    "matched_student_id": alert.matched_student_id,
                    "similarity_score": alert.similarity_score,
                    "detection_method": alert.detection_method,
                    "alert_level": alert.alert_level,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in alerts
            ]
        
        return {
            "students_analyzed": len(student_alerts),
            "total_alerts": sum(len(alerts) for alerts in student_alerts.values()),
            "student_alerts": serialized_alerts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in cross-student analysis: {str(e)}")

@router.post("/search-similar", response_model=List[Dict[str, Any]])
async def search_similar_submissions(
    request: SearchSimilarRequest,
    current_user: User = Depends(get_current_user)
):
    """Search for similar submissions using vector database."""
    try:
        results = await duplicate_detection_service.search_similar_submissions(
            query_text=request.query_text,
            k=request.k,
            threshold=request.threshold
        )
        
        return [
            {
                "submission_id": submission_id,
                "similarity_score": similarity_score,
                "metadata": metadata
            }
            for submission_id, similarity_score, metadata in results
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching similar submissions: {str(e)}")

@router.post("/set-detection-level")
async def set_detection_level(
    request: DetectionLevelRequest,
    current_user: User = Depends(get_current_user)
):
    """Set the detection sensitivity level."""
    try:
        level_map = {
            "strict": DetectionLevel.STRICT,
            "moderate": DetectionLevel.MODERATE,
            "sensitive": DetectionLevel.SENSITIVE
        }
        
        if request.level not in level_map:
            raise HTTPException(
                status_code=400, 
                detail="Invalid detection level. Use: strict, moderate, or sensitive"
            )
        
        duplicate_detection_service.set_detection_level(level_map[request.level])
        
        return {
            "message": f"Detection level set to {request.level}",
            "current_thresholds": duplicate_detection_service.detection_thresholds[level_map[request.level]]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting detection level: {str(e)}")

@router.get("/statistics")
async def get_detection_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get detection system statistics."""
    try:
        stats = duplicate_detection_service.get_detection_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@router.post("/initialize-models")
async def initialize_models(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Initialize embedding models (can be slow, runs in background)."""
    try:
        # Run model initialization in background
        background_tasks.add_task(embedding_service.initialize_models)
        
        return {
            "message": "Model initialization started in background",
            "model_name": embedding_service.model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing models: {str(e)}")

@router.delete("/clear-cache")
async def clear_detection_cache(
    current_user: User = Depends(get_current_user)
):
    """Clear all detection cache and stored embeddings."""
    try:
        await duplicate_detection_service.clear_all_data()
        
        return {"message": "Detection cache cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@router.post("/batch-similarity-search", response_model=Dict[str, Any])
async def batch_similarity_search(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Perform batch similarity search for multiple texts."""
    try:
        query_texts = request.get("query_texts", [])
        candidate_texts = request.get("candidate_texts", {})
        batch_size = request.get("batch_size", 32)
        
        if not query_texts or not candidate_texts:
            raise HTTPException(
                status_code=400,
                detail="Both query_texts and candidate_texts are required"
            )
        
        results = await similarity_service.efficient_batch_similarity_search(
            query_texts=query_texts,
            candidate_texts=candidate_texts,
            batch_size=batch_size
        )
        
        # Convert to serializable format
        serialized_results = {}
        for query_id, similarity_results in results.items():
            serialized_results[query_id] = [
                {
                    "text1_id": result.text1_id,
                    "text2_id": result.text2_id,
                    "similarity_score": result.similarity_score,
                    "is_duplicate": result.is_duplicate,
                    "detection_method": result.detection_method,
                    "metadata": result.metadata or {}
                }
                for result in similarity_results
            ]
        
        return {
            "total_queries": len(query_texts),
            "total_candidates": len(candidate_texts),
            "results": serialized_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch similarity search failed: {str(e)}")

@router.post("/detect-cross-student-duplicates", response_model=List[Dict[str, Any]])
async def detect_cross_student_duplicates(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Detect duplicates across different students."""
    try:
        submissions = request.get("submissions", {})
        similarity_threshold = request.get("similarity_threshold", 0.85)
        
        if not submissions:
            raise HTTPException(status_code=400, detail="Submissions data is required")
        
        duplicates = await duplicate_detection_service.detect_cross_student_duplicates(
            submissions=submissions,
            similarity_threshold=similarity_threshold
        )
        
        return duplicates
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-student duplicate detection failed: {str(e)}")

@router.post("/enhanced-plagiarism-detection", response_model=Dict[str, Any])
async def enhanced_plagiarism_detection(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Enhanced cross-student plagiarism detection with batch processing."""
    try:
        submissions = request.get("submissions", {})
        plagiarism_threshold = request.get("plagiarism_threshold", 0.8)
        batch_processing = request.get("batch_processing", True)
        
        if not submissions:
            raise HTTPException(status_code=400, detail="Submissions data is required")
        
        results = await similarity_service.cross_student_plagiarism_detection(
            submissions=submissions,
            plagiarism_threshold=plagiarism_threshold,
            batch_processing=batch_processing
        )
        
        # Convert to serializable format
        serialized_results = {}
        total_alerts = 0
        
        for student_id, plagiarism_results in results.items():
            serialized_results[student_id] = [
                {
                    "text1_id": result.text1_id,
                    "text2_id": result.text2_id,
                    "similarity_score": result.similarity_score,
                    "is_duplicate": result.is_duplicate,
                    "detection_method": result.detection_method,
                    "metadata": result.metadata or {}
                }
                for result in plagiarism_results
            ]
            total_alerts += len(plagiarism_results)
        
        return {
            "students_analyzed": len(results),
            "total_plagiarism_alerts": total_alerts,
            "plagiarism_threshold": plagiarism_threshold,
            "batch_processing_used": batch_processing,
            "results": serialized_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced plagiarism detection failed: {str(e)}")

@router.get("/vector-database-stats")
async def get_vector_database_stats(
    current_user: User = Depends(get_current_user)
):
    """Get vector database statistics."""
    try:
        from services.vector_database import vector_db
        
        stats = vector_db.get_statistics()
        
        return {
            "vector_database_stats": stats,
            "cache_info": {
                "total_records": len(vector_db.records),
                "is_indexed": vector_db.is_indexed,
                "index_type": vector_db.index_type
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vector database stats: {str(e)}")

@router.get("/health")
async def similarity_health_check():
    """Health check for similarity services."""
    try:
        # Check if models are loaded
        models_loaded = embedding_service.sentence_model is not None
        
        # Get basic stats
        vector_stats = duplicate_detection_service.get_detection_statistics()
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "embedding_model": embedding_service.model_name,
            "vector_database_size": vector_stats["vector_database"]["total_vectors"],
            "detection_level": vector_stats["detection_level"]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
from services.similarity_service import similarity_service
from services.embedding_service import embedding_service

router = APIRouter()

# Pydantic models for API responses
class SimilarityMatchResponse(BaseModel):
    submission_id: str
    student_id: str
    similarity_score: float
    match_type: str
    content_hash: Optional[str] = None
    course_name: Optional[str] = None
    submission_date: Optional[str] = None

class DuplicateDetectionResponse(BaseModel):
    submission_id: str
    is_duplicate: bool
    highest_similarity: float
    matches: List[SimilarityMatchResponse]
    flagged_for_review: bool
    detection_timestamp: str

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity_score: float
    calculation_time_ms: float

class BatchSimilarityRequest(BaseModel):
    submission_ids: List[str]

class EmbeddingRequest(BaseModel):
    text: str
    model_name: Optional[str] = "sentence-bert"

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_name: str
    embedding_dimension: int

@router.post("/calculate-similarity", response_model=SimilarityResponse)
async def calculate_similarity(
    request: SimilarityRequest,
    current_user: User = Depends(get_current_user)
):
    """Calculate cosine similarity between two texts."""
    try:
        import time
        start_time = time.time()
        
        # Generate embeddings for both texts
        embedding1 = await embedding_service.generate_embedding(request.text1)
        embedding2 = await embedding_service.generate_embedding(request.text2)
        
        # Calculate similarity
        similarity = await similarity_service.calculate_cosine_similarity(
            embedding1, embedding2
        )
        
        calculation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SimilarityResponse(
            similarity_score=similarity,
            calculation_time_ms=calculation_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity calculation failed: {str(e)}"
        )

@router.post("/detect-duplicates/{submission_id}", response_model=DuplicateDetectionResponse)
async def detect_duplicates(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Detect duplicates for a specific submission."""
    try:
        # Get submission from database
        from sqlalchemy import select
        from models.submission import Submission
        
        query = select(Submission).where(Submission.id == int(submission_id))
        result = await db.execute(query)
        submission = result.scalar_one_or_none()
        
        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found"
            )
        
        # Check if user has permission to access this submission
        if current_user.role != "admin" and submission.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this submission"
            )
        
        # Perform duplicate detection
        detection_result = await similarity_service.detect_duplicates(
            submission_id=str(submission.id),
            content=submission.content,
            student_id=str(submission.user_id),
            db=db,
            content_hash=submission.file_hash
        )
        
        # Convert to response format
        matches = [
            SimilarityMatchResponse(
                submission_id=match.submission_id,
                student_id=match.student_id,
                similarity_score=match.similarity_score,
                match_type=match.match_type.value,
                content_hash=match.content_hash,
                course_name=match.course_name,
                submission_date=match.submission_date.isoformat() if match.submission_date else None
            )
            for match in detection_result.matches
        ]
        
        return DuplicateDetectionResponse(
            submission_id=detection_result.submission_id,
            is_duplicate=detection_result.is_duplicate,
            highest_similarity=detection_result.highest_similarity,
            matches=matches,
            flagged_for_review=detection_result.flagged_for_review,
            detection_timestamp=detection_result.detection_timestamp.isoformat()
        )
        
    except DuplicateDetectionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Duplicate detection failed: {str(e)}"
        )

@router.post("/batch-similarity-analysis", response_model=Dict[str, DuplicateDetectionResponse])
async def batch_similarity_analysis(
    request: BatchSimilarityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Perform similarity analysis on multiple submissions."""
    # Only admins can perform batch analysis
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for batch analysis"
        )
    
    try:
        results = await similarity_service.batch_similarity_analysis(
            request.submission_ids, db
        )
        
        # Convert to response format
        response = {}
        for submission_id, detection_result in results.items():
            matches = [
                SimilarityMatchResponse(
                    submission_id=match.submission_id,
                    student_id=match.student_id,
                    similarity_score=match.similarity_score,
                    match_type=match.match_type.value,
                    content_hash=match.content_hash,
                    course_name=match.course_name,
                    submission_date=match.submission_date.isoformat() if match.submission_date else None
                )
                for match in detection_result.matches
            ]
            
            response[submission_id] = DuplicateDetectionResponse(
                submission_id=detection_result.submission_id,
                is_duplicate=detection_result.is_duplicate,
                highest_similarity=detection_result.highest_similarity,
                matches=matches,
                flagged_for_review=detection_result.flagged_for_review,
                detection_timestamp=detection_result.detection_timestamp.isoformat()
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )

@router.get("/cross-student-plagiarism/{student_id}")
async def detect_cross_student_plagiarism(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Detect cross-student plagiarism for a specific student."""
    # Only admins or the student themselves can access this
    if current_user.role != "admin" and str(current_user.id) != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this data"
        )
    
    try:
        plagiarism_cases = await similarity_service.detect_cross_student_plagiarism(
            student_id, db
        )
        
        return {
            "student_id": student_id,
            "plagiarism_cases": plagiarism_cases,
            "total_cases": len(plagiarism_cases)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plagiarism detection failed: {str(e)}"
        )

@router.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate embedding for text."""
    try:
        embedding = await embedding_service.generate_embedding(
            request.text, request.model_name
        )
        
        model_info = await embedding_service.get_embedding_info(request.model_name)
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            model_name=request.model_name,
            embedding_dimension=model_info['embedding_dimension']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )

@router.get("/similarity-statistics")
async def get_similarity_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get similarity detection statistics."""
    # Only admins can access statistics
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        stats = await similarity_service.get_similarity_statistics()
        embedding_info = await embedding_service.get_embedding_info()
        cache_stats = embedding_service.get_cache_stats()
        
        return {
            "similarity_stats": stats,
            "embedding_info": embedding_info,
            "cache_stats": cache_stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )

@router.post("/update-thresholds")
async def update_similarity_thresholds(
    exact_duplicate: Optional[float] = None,
    near_duplicate: Optional[float] = None,
    semantic_similarity: Optional[float] = None,
    review_threshold: Optional[float] = None,
    current_user: User = Depends(get_current_user)
):
    """Update similarity detection thresholds."""
    # Only admins can update thresholds
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        await similarity_service.update_thresholds(
            exact_duplicate=exact_duplicate,
            near_duplicate=near_duplicate,
            semantic_similarity=semantic_similarity,
            review_threshold=review_threshold
        )
        
        return {"message": "Thresholds updated successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update thresholds: {str(e)}"
        )
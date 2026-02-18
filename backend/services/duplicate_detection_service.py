"""
Comprehensive duplicate detection service integrating semantic similarity,
vector search, and cross-student plagiarism detection.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime

from .embedding_service import embedding_service
from .similarity_service import similarity_service, SimilarityResult, DuplicateDetectionResult
from .vector_database import vector_db

logger = logging.getLogger(__name__)

class DetectionLevel(Enum):
    """Detection sensitivity levels."""
    STRICT = "strict"      # High thresholds, fewer false positives
    MODERATE = "moderate"  # Balanced thresholds
    SENSITIVE = "sensitive" # Lower thresholds, catches more potential cases

@dataclass
class PlagiarismAlert:
    """Alert for potential plagiarism detection."""
    submission_id: str
    student_id: str
    matched_submission_id: str
    matched_student_id: str
    similarity_score: float
    detection_method: str
    alert_level: str  # "high", "medium", "low"
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class DuplicateAnalysisReport:
    """Comprehensive report of duplicate analysis."""
    query_submission_id: str
    total_comparisons: int
    duplicates_found: int
    potential_duplicates: int
    highest_similarity: float
    processing_time: float
    alerts: List[PlagiarismAlert]
    detailed_results: List[SimilarityResult]

class DuplicateDetectionService:
    """Main service for comprehensive duplicate and plagiarism detection."""
    
    def __init__(self):
        """Initialize the duplicate detection service."""
        self.detection_thresholds = {
            DetectionLevel.STRICT: {"semantic": 0.90, "lexical": 0.85},
            DetectionLevel.MODERATE: {"semantic": 0.80, "lexical": 0.75},
            DetectionLevel.SENSITIVE: {"semantic": 0.70, "lexical": 0.65}
        }
        
        self.current_level = DetectionLevel.MODERATE
        self._update_service_thresholds()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the duplicate detection service (async initialization)."""
        if self.initialized:
            return
        
        try:
            # Initialize vector database if needed
            # await vector_db.initialize()  # Uncomment if vector_db has initialize method
            self.initialized = True
            logger.info("Duplicate detection service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing duplicate detection service: {e}")
            # Continue anyway with basic functionality
            self.initialized = True
    
    def _update_service_thresholds(self):
        """Update similarity service thresholds based on current detection level."""
        thresholds = self.detection_thresholds[self.current_level]
        similarity_service.update_thresholds(
            semantic_threshold=thresholds["semantic"],
            lexical_threshold=thresholds["lexical"]
        )
    
    def set_detection_level(self, level: DetectionLevel):
        """Set the detection sensitivity level."""
        self.current_level = level
        self._update_service_thresholds()
        logger.info(f"Detection level set to {level.value}")
    
    async def analyze_submission(self, 
                               submission_id: str,
                               submission_text: str,
                               student_id: str,
                               existing_submissions: Dict[str, Dict[str, str]],
                               exclude_same_student: bool = True) -> DuplicateAnalysisReport:
        """
        Analyze a submission for duplicates against existing submissions.
        
        Args:
            submission_id: ID of the submission to analyze
            submission_text: Text content of the submission
            student_id: ID of the student who submitted
            existing_submissions: Dict of {sub_id: {"text": content, "student_id": id}}
            exclude_same_student: Whether to exclude same student's submissions
            
        Returns:
            Comprehensive analysis report
        """
        import time
        start_time = time.time()
        
        try:
            # Filter existing submissions
            candidates = {}
            for sub_id, sub_data in existing_submissions.items():
                if sub_id == submission_id:
                    continue
                
                if exclude_same_student and sub_data.get("student_id") == student_id:
                    continue
                
                candidates[sub_id] = sub_data["text"]
            
            # Perform duplicate detection
            detection_result = await similarity_service.detect_duplicates(
                query_text=submission_text,
                candidate_texts=candidates,
                query_text_id=submission_id,
                use_semantic=True,
                use_lexical=True
            )
            
            # Generate alerts
            alerts = self._generate_alerts(
                submission_id, student_id, detection_result, existing_submissions
            )
            
            # Calculate statistics
            highest_similarity = 0.0
            if detection_result.duplicates:
                highest_similarity = max(r.similarity_score for r in detection_result.duplicates)
            elif detection_result.potential_duplicates:
                highest_similarity = max(r.similarity_score for r in detection_result.potential_duplicates)
            
            processing_time = time.time() - start_time
            
            # Create comprehensive report
            report = DuplicateAnalysisReport(
                query_submission_id=submission_id,
                total_comparisons=detection_result.total_comparisons,
                duplicates_found=len(detection_result.duplicates),
                potential_duplicates=len(detection_result.potential_duplicates),
                highest_similarity=highest_similarity,
                processing_time=processing_time,
                alerts=alerts,
                detailed_results=detection_result.duplicates + detection_result.potential_duplicates
            )
            
            # Store in vector database for future comparisons
            await self._store_submission_embedding(submission_id, submission_text, {
                "student_id": student_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_performed": True
            })
            
            logger.info(f"Analysis complete for {submission_id}: {len(alerts)} alerts generated")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing submission {submission_id}: {e}")
            raise
    
    def _generate_alerts(self, 
                        submission_id: str,
                        student_id: str,
                        detection_result: DuplicateDetectionResult,
                        existing_submissions: Dict[str, Dict[str, str]]) -> List[PlagiarismAlert]:
        """Generate plagiarism alerts based on detection results."""
        alerts = []
        
        # High-priority alerts for confirmed duplicates
        for result in detection_result.duplicates:
            matched_sub_data = existing_submissions.get(result.text2_id, {})
            
            alert_level = "high"
            if result.similarity_score >= 0.95:
                alert_level = "high"
            elif result.similarity_score >= 0.85:
                alert_level = "medium"
            else:
                alert_level = "low"
            
            alert = PlagiarismAlert(
                submission_id=submission_id,
                student_id=student_id,
                matched_submission_id=result.text2_id,
                matched_student_id=matched_sub_data.get("student_id", "unknown"),
                similarity_score=result.similarity_score,
                detection_method=result.detection_method,
                alert_level=alert_level,
                timestamp=datetime.now(),
                metadata=result.metadata
            )
            alerts.append(alert)
        
        # Medium-priority alerts for potential duplicates
        for result in detection_result.potential_duplicates:
            if result.similarity_score >= 0.7:  # Only alert on higher potential matches
                matched_sub_data = existing_submissions.get(result.text2_id, {})
                
                alert = PlagiarismAlert(
                    submission_id=submission_id,
                    student_id=student_id,
                    matched_submission_id=result.text2_id,
                    matched_student_id=matched_sub_data.get("student_id", "unknown"),
                    similarity_score=result.similarity_score,
                    detection_method=result.detection_method,
                    alert_level="low",
                    timestamp=datetime.now(),
                    metadata=result.metadata
                )
                alerts.append(alert)
        
        return alerts
    
    async def batch_analyze_submissions(self, 
                                      submissions: Dict[str, Dict[str, str]],
                                      batch_size: int = 10) -> Dict[str, DuplicateAnalysisReport]:
        """
        Analyze multiple submissions in batches.
        
        Args:
            submissions: Dict of {sub_id: {"text": content, "student_id": id}}
            batch_size: Number of submissions to process concurrently
            
        Returns:
            Dict mapping submission IDs to their analysis reports
        """
        results = {}
        submission_ids = list(submissions.keys())
        
        for i in range(0, len(submission_ids), batch_size):
            batch_ids = submission_ids[i:i + batch_size]
            batch_tasks = []
            
            for sub_id in batch_ids:
                sub_data = submissions[sub_id]
                
                # Create candidates excluding current submission
                candidates = {
                    other_id: other_data 
                    for other_id, other_data in submissions.items() 
                    if other_id != sub_id
                }
                
                task = self.analyze_submission(
                    submission_id=sub_id,
                    submission_text=sub_data["text"],
                    student_id=sub_data["student_id"],
                    existing_submissions=candidates
                )
                batch_tasks.append((sub_id, task))
            
            # Execute batch
            for sub_id, task in batch_tasks:
                try:
                    result = await task
                    results[sub_id] = result
                except Exception as e:
                    logger.error(f"Error processing submission {sub_id}: {e}")
        
        logger.info(f"Batch analysis complete: {len(results)} submissions processed")
        return results
    
    async def cross_student_analysis(self, 
                                   submissions: Dict[str, Dict[str, str]]) -> Dict[str, List[PlagiarismAlert]]:
        """
        Perform comprehensive cross-student plagiarism analysis.
        
        Args:
            submissions: Dict of {sub_id: {"text": content, "student_id": id}}
            
        Returns:
            Dict mapping student IDs to their plagiarism alerts
        """
        try:
            # Use similarity service for cross-student detection
            plagiarism_results = await similarity_service.cross_student_plagiarism_detection(
                submissions=submissions
            )
            
            # Convert to alerts format
            student_alerts = {}
            
            for student_id, similarity_results in plagiarism_results.items():
                alerts = []
                
                for result in similarity_results:
                    # Get matched student info
                    matched_sub_data = submissions.get(result.text2_id, {})
                    
                    alert = PlagiarismAlert(
                        submission_id=result.text1_id,
                        student_id=student_id,
                        matched_submission_id=result.text2_id,
                        matched_student_id=matched_sub_data.get("student_id", "unknown"),
                        similarity_score=result.similarity_score,
                        detection_method=result.detection_method,
                        alert_level="high" if result.similarity_score >= 0.9 else "medium",
                        timestamp=datetime.now(),
                        metadata=result.metadata
                    )
                    alerts.append(alert)
                
                student_alerts[student_id] = alerts
            
            return student_alerts
            
        except Exception as e:
            logger.error(f"Error in cross-student analysis: {e}")
            raise
    
    async def _store_submission_embedding(self, 
                                        submission_id: str,
                                        text: str,
                                        metadata: Dict[str, Any]):
        """Store submission embedding in vector database for future searches."""
        try:
            # Generate embedding
            embeddings = await embedding_service.generate_embeddings([text])
            if len(embeddings) > 0:
                vector_db.add_vector(
                    vector_id=submission_id,
                    vector=embeddings[0],
                    metadata=metadata
                )
                logger.debug(f"Stored embedding for submission {submission_id}")
        except Exception as e:
            logger.warning(f"Could not store embedding for {submission_id}: {e}")
    
    async def search_similar_submissions(self, 
                                       query_text: str,
                                       k: int = 10,
                                       threshold: float = 0.7,
                                       exclude_student_id: str = None,
                                       include_metadata_filter: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Enhanced search for similar submissions using vector database with filtering.
        
        Args:
            query_text: Text to search for
            k: Number of similar submissions to return
            threshold: Minimum similarity threshold
            exclude_student_id: Student ID to exclude from results
            include_metadata_filter: Additional metadata filters to apply
            
        Returns:
            List of (submission_id, similarity_score, metadata) tuples
        """
        try:
            # Generate query embedding
            embeddings = await embedding_service.generate_embeddings([query_text])
            if len(embeddings) == 0:
                return []
            
            # Prepare metadata filter
            metadata_filter = include_metadata_filter or {}
            if exclude_student_id:
                metadata_filter["student_id"] = {"not": exclude_student_id}
            
            # Search vector database with filtering
            results = await vector_db.search_similar(
                query_vector=embeddings[0],
                k=k * 2,  # Search more to account for filtering
                threshold=threshold,
                filter_metadata=metadata_filter if metadata_filter else None
            )
            
            # Limit to requested number of results
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error searching similar submissions: {e}")
            return []
    
    async def find_duplicates(self,
                            text_embedding,
                            text: str,
                            exclude_student_id: str = None,
                            exclude_submission_id: str = None,
                            threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Find duplicate submissions for a given text.
        
        Args:
            text_embedding: Embedding vector of the text
            text: The text content
            exclude_student_id: Student ID to exclude from results
            exclude_submission_id: Submission ID to exclude from results
            threshold: Similarity threshold
            
        Returns:
            List of duplicate matches
        """
        try:
            # For now, return empty list as placeholder
            # In production, this would search the vector database
            logger.info(f"Searching for duplicates (threshold: {threshold})")
            return []
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return []
    
    async def detect_cross_student_duplicates(self, 
                                            submissions: Dict[str, Dict[str, str]],
                                            similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Detect duplicates across different students using efficient vector search.
        
        Args:
            submissions: Dictionary of {submission_id: {"text": content, "student_id": id, ...}}
            similarity_threshold: Threshold for duplicate detection
            
        Returns:
            List of duplicate detection results
        """
        try:
            duplicate_pairs = []
            
            # Use vector database to find duplicates efficiently
            duplicates = await vector_db.find_duplicates(
                similarity_threshold=similarity_threshold,
                exclude_same_metadata={"student_id": "same"}
            )
            
            for sub_id1, sub_id2, similarity in duplicates:
                # Get submission metadata
                metadata1 = vector_db.get_vector(sub_id1)
                metadata2 = vector_db.get_vector(sub_id2)
                
                if metadata1 and metadata2:
                    duplicate_info = {
                        "submission_1": {
                            "id": sub_id1,
                            "student_id": metadata1.metadata.get("student_id"),
                            "timestamp": metadata1.metadata.get("timestamp")
                        },
                        "submission_2": {
                            "id": sub_id2,
                            "student_id": metadata2.metadata.get("student_id"),
                            "timestamp": metadata2.metadata.get("timestamp")
                        },
                        "similarity_score": similarity,
                        "detection_method": "vector_database_cross_student",
                        "flagged_at": datetime.now().isoformat()
                    }
                    duplicate_pairs.append(duplicate_info)
            
            return duplicate_pairs
            
        except Exception as e:
            logger.error(f"Error in cross-student duplicate detection: {e}")
            return []
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about the detection system."""
        vector_stats = vector_db.get_statistics()
        
        return {
            "detection_level": self.current_level.value,
            "thresholds": self.detection_thresholds[self.current_level],
            "vector_database": vector_stats,
            "embedding_model": embedding_service.model_name,
            "embedding_dimension": embedding_service.get_embedding_dimension()
        }
    
    async def clear_all_data(self):
        """Clear all stored embeddings and cache."""
        vector_db.clear_database()
        embedding_service.clear_cache()
        logger.info("Cleared all detection data")


# Global instance
duplicate_detection_service = DuplicateDetectionService()
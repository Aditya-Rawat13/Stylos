"""
Similarity service for duplicate detection and plagiarism analysis.
Handles cosine similarity calculations, vector search, and duplicate flagging.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .embedding_service import embedding_service

logger = logging.getLogger(__name__)

class SimilarityThreshold(Enum):
    """Predefined similarity thresholds for different detection levels."""
    LOW = 0.6
    MEDIUM = 0.75
    HIGH = 0.85
    VERY_HIGH = 0.95

@dataclass
class SimilarityResult:
    """Result of similarity comparison between two texts."""
    text1_id: str
    text2_id: str
    similarity_score: float
    is_duplicate: bool
    detection_method: str
    metadata: Dict = None

@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection analysis."""
    query_text_id: str
    duplicates: List[SimilarityResult]
    potential_duplicates: List[SimilarityResult]
    total_comparisons: int
    processing_time: float

class SimilarityService:
    """Service for detecting duplicates and measuring text similarity."""
    
    def __init__(self, 
                 semantic_threshold: float = SimilarityThreshold.HIGH.value,
                 lexical_threshold: float = SimilarityThreshold.MEDIUM.value):
        """
        Initialize similarity service.
        
        Args:
            semantic_threshold: Threshold for semantic similarity detection
            lexical_threshold: Threshold for lexical similarity detection
        """
        self.semantic_threshold = semantic_threshold
        self.lexical_threshold = lexical_threshold
        
        # TF-IDF vectorizer for lexical similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        
        # Storage for text embeddings and metadata
        self.text_embeddings = {}
        self.text_metadata = {}
        
        # Vector database simulation (in production, use proper vector DB)
        self.vector_index = {}
    
    async def calculate_semantic_similarity(self, 
                                          text1: str, 
                                          text2: str,
                                          text1_id: str = None,
                                          text2_id: str = None) -> SimilarityResult:
        """
        Calculate semantic similarity between two texts using embeddings.
        
        Args:
            text1: First text
            text2: Second text
            text1_id: Optional ID for first text
            text2_id: Optional ID for second text
            
        Returns:
            SimilarityResult with semantic similarity score
        """
        try:
            # Generate embeddings
            embeddings = await embedding_service.generate_embeddings([text1, text2])
            
            if len(embeddings) != 2:
                raise ValueError("Failed to generate embeddings for both texts")
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
            similarity_score = float(similarity_matrix[0][0])
            
            is_duplicate = similarity_score >= self.semantic_threshold
            
            return SimilarityResult(
                text1_id=text1_id or "text1",
                text2_id=text2_id or "text2",
                similarity_score=similarity_score,
                is_duplicate=is_duplicate,
                detection_method="semantic_embedding",
                metadata={
                    "embedding_model": getattr(embedding_service, 'model_name', 'unknown'),
                    "threshold_used": self.semantic_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            raise
    
    def calculate_lexical_similarity(self, 
                                   text1: str, 
                                   text2: str,
                                   text1_id: str = None,
                                   text2_id: str = None) -> SimilarityResult:
        """
        Calculate lexical similarity using TF-IDF vectors.
        
        Args:
            text1: First text
            text2: Second text
            text1_id: Optional ID for first text
            text2_id: Optional ID for second text
            
        Returns:
            SimilarityResult with lexical similarity score
        """
        try:
            # Fit TF-IDF on both texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity_score = float(similarity_matrix[0][0])
            
            is_duplicate = similarity_score >= self.lexical_threshold
            
            return SimilarityResult(
                text1_id=text1_id or "text1",
                text2_id=text2_id or "text2",
                similarity_score=similarity_score,
                is_duplicate=is_duplicate,
                detection_method="lexical_tfidf",
                metadata={
                    "threshold_used": self.lexical_threshold,
                    "features_used": len(self.tfidf_vectorizer.get_feature_names_out())
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating lexical similarity: {e}")
            raise
    
    async def detect_duplicates(self, 
                              query_text: str,
                              candidate_texts: Dict[str, str],
                              query_text_id: str = "query",
                              use_semantic: bool = True,
                              use_lexical: bool = True,
                              custom_thresholds: Dict[str, float] = None) -> DuplicateDetectionResult:
        """
        Detect duplicates of query text among candidate texts with enhanced threshold-based flagging.
        
        Args:
            query_text: Text to check for duplicates
            candidate_texts: Dictionary of {text_id: text_content} to search
            query_text_id: ID for the query text
            use_semantic: Whether to use semantic similarity
            use_lexical: Whether to use lexical similarity
            custom_thresholds: Optional custom thresholds for detection
            
        Returns:
            DuplicateDetectionResult with found duplicates
        """
        import time
        start_time = time.time()
        
        duplicates = []
        potential_duplicates = []
        total_comparisons = len(candidate_texts)
        
        # Use custom thresholds if provided
        thresholds = custom_thresholds or {
            "duplicate": max(self.semantic_threshold, self.lexical_threshold),
            "potential": 0.5,
            "semantic": self.semantic_threshold,
            "lexical": self.lexical_threshold
        }
        
        try:
            # Generate query embedding once for efficiency
            query_embedding = None
            if use_semantic:
                query_embedding = await embedding_service.generate_embedding(query_text)
            
            for candidate_id, candidate_text in candidate_texts.items():
                # Skip self-comparison
                if candidate_id == query_text_id:
                    continue
                
                results = []
                max_similarity = 0.0
                
                # Semantic similarity check with optimized embedding reuse
                if use_semantic and query_embedding is not None:
                    candidate_embedding = await embedding_service.generate_embedding(candidate_text)
                    semantic_similarity = await self._calculate_cosine_similarity_optimized(
                        query_embedding, candidate_embedding
                    )
                    
                    semantic_result = SimilarityResult(
                        text1_id=query_text_id,
                        text2_id=candidate_id,
                        similarity_score=semantic_similarity,
                        is_duplicate=semantic_similarity >= thresholds["semantic"],
                        detection_method="semantic_embedding",
                        metadata={
                            "threshold_used": thresholds["semantic"],
                            "embedding_model": getattr(embedding_service, 'model_name', 'unknown')
                        }
                    )
                    results.append(semantic_result)
                    max_similarity = max(max_similarity, semantic_similarity)
                
                # Lexical similarity check
                if use_lexical:
                    lexical_result = self.calculate_lexical_similarity(
                        query_text, candidate_text, query_text_id, candidate_id
                    )
                    # Update threshold in metadata
                    lexical_result.metadata["threshold_used"] = thresholds["lexical"]
                    results.append(lexical_result)
                    max_similarity = max(max_similarity, lexical_result.similarity_score)
                
                # Enhanced duplicate determination with multiple criteria
                is_duplicate = self._determine_duplicate_status(results, thresholds)
                
                # Create combined result with enhanced metadata
                combined_result = SimilarityResult(
                    text1_id=query_text_id,
                    text2_id=candidate_id,
                    similarity_score=max_similarity,
                    is_duplicate=is_duplicate,
                    detection_method="combined_threshold_based",
                    metadata={
                        "semantic_score": next((r.similarity_score for r in results if r.detection_method == "semantic_embedding"), None),
                        "lexical_score": next((r.similarity_score for r in results if r.detection_method == "lexical_tfidf"), None),
                        "methods_used": [r.detection_method for r in results],
                        "thresholds_applied": thresholds,
                        "flagging_reason": self._get_flagging_reason(results, thresholds),
                        "confidence_level": self._calculate_confidence_level(results, max_similarity)
                    }
                )
                
                # Enhanced categorization with threshold-based flagging
                if is_duplicate:
                    duplicates.append(combined_result)
                elif max_similarity >= thresholds["potential"]:
                    potential_duplicates.append(combined_result)
            
            # Sort by similarity score (descending)
            duplicates.sort(key=lambda x: x.similarity_score, reverse=True)
            potential_duplicates.sort(key=lambda x: x.similarity_score, reverse=True)
            
            processing_time = time.time() - start_time
            
            return DuplicateDetectionResult(
                query_text_id=query_text_id,
                duplicates=duplicates,
                potential_duplicates=potential_duplicates,
                total_comparisons=total_comparisons,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            raise
    
    async def cross_student_plagiarism_detection(self, 
                                               submissions: Dict[str, Dict[str, str]],
                                               student_groups: Dict[str, List[str]] = None,
                                               plagiarism_threshold: float = 0.8,
                                               batch_processing: bool = True) -> Dict[str, List[SimilarityResult]]:
        """
        Enhanced cross-student plagiarism detection with efficient processing.
        
        Args:
            submissions: Dictionary of {submission_id: {"text": content, "student_id": id, ...}}
            student_groups: Optional grouping of students to check within groups
            plagiarism_threshold: Threshold for flagging potential plagiarism
            batch_processing: Whether to use batch processing for efficiency
            
        Returns:
            Dictionary mapping student IDs to their plagiarism results
        """
        plagiarism_results = {}
        
        try:
            # Group submissions by student if not provided
            if student_groups is None:
                student_submissions = {}
                for sub_id, sub_data in submissions.items():
                    student_id = sub_data.get("student_id")
                    if student_id:
                        if student_id not in student_submissions:
                            student_submissions[student_id] = []
                        student_submissions[student_id].append(sub_id)
                student_groups = student_submissions
            
            # Use batch processing for efficiency if enabled
            if batch_processing and len(submissions) > 10:
                return await self._batch_cross_student_detection(
                    submissions, student_groups, plagiarism_threshold
                )
            
            # Check each student's submissions against others
            for student_id, student_sub_ids in student_groups.items():
                student_results = []
                
                for sub_id in student_sub_ids:
                    if sub_id not in submissions:
                        continue
                    
                    query_text = submissions[sub_id]["text"]
                    
                    # Create candidate texts (excluding same student's submissions)
                    candidates = {
                        other_sub_id: other_sub_data["text"]
                        for other_sub_id, other_sub_data in submissions.items()
                        if other_sub_data.get("student_id") != student_id
                    }
                    
                    if candidates:
                        # Use custom threshold for plagiarism detection
                        custom_thresholds = {
                            "duplicate": plagiarism_threshold,
                            "potential": plagiarism_threshold * 0.7,
                            "semantic": plagiarism_threshold,
                            "lexical": plagiarism_threshold * 0.8
                        }
                        
                        detection_result = await self.detect_duplicates(
                            query_text=query_text,
                            candidate_texts=candidates,
                            query_text_id=sub_id,
                            use_semantic=True,
                            use_lexical=True,
                            custom_thresholds=custom_thresholds
                        )
                        
                        # Add all duplicates and high-similarity potential duplicates
                        for result in detection_result.duplicates:
                            # Enhance metadata for plagiarism context
                            result.metadata.update({
                                "plagiarism_detection": True,
                                "source_student": student_id,
                                "matched_student": submissions[result.text2_id].get("student_id"),
                                "detection_context": "cross_student_plagiarism"
                            })
                            student_results.append(result)
                        
                        # Include high-confidence potential duplicates
                        for result in detection_result.potential_duplicates:
                            if result.similarity_score >= plagiarism_threshold * 0.8:
                                result.metadata.update({
                                    "plagiarism_detection": True,
                                    "source_student": student_id,
                                    "matched_student": submissions[result.text2_id].get("student_id"),
                                    "detection_context": "potential_cross_student_plagiarism"
                                })
                                student_results.append(result)
                
                # Sort by similarity score (highest first)
                student_results.sort(key=lambda x: x.similarity_score, reverse=True)
                plagiarism_results[student_id] = student_results
            
            return plagiarism_results
            
        except Exception as e:
            logger.error(f"Error in cross-student plagiarism detection: {e}")
            raise
    
    async def _batch_cross_student_detection(self, 
                                           submissions: Dict[str, Dict[str, str]],
                                           student_groups: Dict[str, List[str]],
                                           plagiarism_threshold: float) -> Dict[str, List[SimilarityResult]]:
        """
        Batch processing for cross-student plagiarism detection.
        
        Args:
            submissions: Dictionary of submissions
            student_groups: Student groupings
            plagiarism_threshold: Threshold for plagiarism detection
            
        Returns:
            Dictionary mapping student IDs to plagiarism results
        """
        plagiarism_results = {}
        
        try:
            # Create submission text mapping
            submission_texts = {sub_id: sub_data["text"] for sub_id, sub_data in submissions.items()}
            
            # Build similarity matrix for all submissions
            similarity_matrix, submission_ids = await self.build_similarity_matrix(submission_texts)
            
            # Process results for each student
            for student_id, student_sub_ids in student_groups.items():
                student_results = []
                
                for sub_id in student_sub_ids:
                    if sub_id not in submission_ids:
                        continue
                    
                    sub_index = submission_ids.index(sub_id)
                    
                    # Check similarities with other students' submissions
                    for other_index, other_sub_id in enumerate(submission_ids):
                        if other_sub_id == sub_id:
                            continue
                        
                        other_student_id = submissions[other_sub_id].get("student_id")
                        if other_student_id == student_id:
                            continue  # Skip same student
                        
                        similarity_score = float(similarity_matrix[sub_index][other_index])
                        
                        if similarity_score >= plagiarism_threshold * 0.7:  # Include potential matches
                            is_duplicate = similarity_score >= plagiarism_threshold
                            
                            result = SimilarityResult(
                                text1_id=sub_id,
                                text2_id=other_sub_id,
                                similarity_score=similarity_score,
                                is_duplicate=is_duplicate,
                                detection_method="batch_cross_student_semantic",
                                metadata={
                                    "plagiarism_detection": True,
                                    "source_student": student_id,
                                    "matched_student": other_student_id,
                                    "detection_context": "batch_cross_student_plagiarism",
                                    "threshold_used": plagiarism_threshold,
                                    "batch_processed": True
                                }
                            )
                            student_results.append(result)
                
                # Sort by similarity score (highest first)
                student_results.sort(key=lambda x: x.similarity_score, reverse=True)
                plagiarism_results[student_id] = student_results
            
            return plagiarism_results
            
        except Exception as e:
            logger.error(f"Error in batch cross-student detection: {e}")
            raise
    
    async def build_similarity_matrix(self, texts: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Build a similarity matrix for all texts.
        
        Args:
            texts: Dictionary of {text_id: text_content}
            
        Returns:
            Tuple of (similarity_matrix, text_ids_list)
        """
        text_ids = list(texts.keys())
        text_contents = list(texts.values())
        
        if not text_contents:
            return np.array([]), []
        
        try:
            # Generate embeddings for all texts
            embeddings = await embedding_service.generate_embeddings(text_contents)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)
            
            return similarity_matrix, text_ids
            
        except Exception as e:
            logger.error(f"Error building similarity matrix: {e}")
            raise
    
    def find_similar_texts(self, 
                          similarity_matrix: np.ndarray, 
                          text_ids: List[str],
                          threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Find pairs of similar texts from similarity matrix.
        
        Args:
            similarity_matrix: Precomputed similarity matrix
            text_ids: List of text IDs corresponding to matrix indices
            threshold: Similarity threshold (uses semantic_threshold if None)
            
        Returns:
            List of (text1_id, text2_id, similarity_score) tuples
        """
        if threshold is None:
            threshold = self.semantic_threshold
        
        similar_pairs = []
        n = len(text_ids)
        
        # Check upper triangle of matrix (avoid duplicates and self-comparisons)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    similar_pairs.append((text_ids[i], text_ids[j], float(similarity)))
        
        # Sort by similarity score (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    async def _calculate_cosine_similarity_optimized(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Optimized cosine similarity calculation for embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Normalize embeddings for efficiency
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity using dot product
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in valid range [0, 1] and handle numerical precision
            similarity = max(0.0, min(1.0, float(similarity)))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error in optimized cosine similarity calculation: {e}")
            return 0.0
    
    def _determine_duplicate_status(self, results: List[SimilarityResult], thresholds: Dict[str, float]) -> bool:
        """
        Determine if texts are duplicates based on multiple similarity results and thresholds.
        
        Args:
            results: List of similarity results from different methods
            thresholds: Dictionary of threshold values
            
        Returns:
            True if considered duplicate, False otherwise
        """
        if not results:
            return False
        
        # Check if any method indicates duplicate
        method_duplicates = [result.is_duplicate for result in results]
        
        # Get maximum similarity score
        max_similarity = max(result.similarity_score for result in results)
        
        # Enhanced logic: duplicate if any method exceeds its threshold OR
        # if maximum similarity exceeds the general duplicate threshold
        return any(method_duplicates) or max_similarity >= thresholds.get("duplicate", 0.85)
    
    def _get_flagging_reason(self, results: List[SimilarityResult], thresholds: Dict[str, float]) -> str:
        """
        Get the reason why a text pair was flagged as duplicate or potential duplicate.
        
        Args:
            results: List of similarity results
            thresholds: Dictionary of threshold values
            
        Returns:
            String describing the flagging reason
        """
        if not results:
            return "no_analysis"
        
        reasons = []
        
        for result in results:
            if result.is_duplicate:
                reasons.append(f"{result.detection_method}_threshold_exceeded")
        
        max_similarity = max(result.similarity_score for result in results)
        if max_similarity >= thresholds.get("duplicate", 0.85):
            reasons.append("high_overall_similarity")
        elif max_similarity >= thresholds.get("potential", 0.5):
            reasons.append("moderate_similarity")
        
        return "; ".join(reasons) if reasons else "low_similarity"
    
    def _calculate_confidence_level(self, results: List[SimilarityResult], max_similarity: float) -> str:
        """
        Calculate confidence level for the duplicate detection result.
        
        Args:
            results: List of similarity results
            max_similarity: Maximum similarity score
            
        Returns:
            Confidence level as string
        """
        if max_similarity >= 0.95:
            return "very_high"
        elif max_similarity >= 0.85:
            return "high"
        elif max_similarity >= 0.7:
            return "medium"
        elif max_similarity >= 0.5:
            return "low"
        else:
            return "very_low"
    
    def update_thresholds(self, semantic_threshold: float = None, lexical_threshold: float = None):
        """Update similarity thresholds."""
        if semantic_threshold is not None:
            self.semantic_threshold = semantic_threshold
        if lexical_threshold is not None:
            self.lexical_threshold = lexical_threshold
        
        logger.info(f"Updated thresholds - Semantic: {self.semantic_threshold}, Lexical: {self.lexical_threshold}")


# Global instance
similarity_service = SimilarityService()
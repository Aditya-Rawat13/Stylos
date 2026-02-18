"""
Tests for semantic embedding and similarity systems.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch

from ..services.embedding_service import EmbeddingService
from ..services.similarity_service import SimilarityService
from ..services.vector_database import VectorDatabase
from ..services.duplicate_detection_service import DuplicateDetectionService, DetectionLevel

class TestEmbeddingService:
    """Test cases for embedding service."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance for testing."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, embedding_service):
        """Test embedding generation."""
        texts = [
            "This is a test sentence.",
            "This is another test sentence with similar content.",
            "Completely different content about cats and dogs."
        ]
        
        # Mock the sentence transformer to avoid downloading models in tests
        with patch.object(embedding_service, '_generate_sentence_embeddings') as mock_embed:
            mock_embed.return_value = np.random.rand(3, 384)  # Mock embeddings
            
            embeddings = await embedding_service.generate_embeddings(texts)
            
            assert embeddings.shape == (3, 384)
            assert isinstance(embeddings, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, embedding_service):
        """Test batch processing of texts."""
        texts = [f"Test sentence {i}" for i in range(10)]
        
        with patch.object(embedding_service, 'generate_embeddings') as mock_embed:
            mock_embed.return_value = np.random.rand(5, 384)  # Mock batch embeddings
            
            embeddings = await embedding_service.process_batch(texts, batch_size=5)
            
            # Should be called twice for 10 texts with batch_size=5
            assert mock_embed.call_count == 2
    
    def test_embedding_storage(self, embedding_service):
        """Test embedding storage and retrieval."""
        text_ids = ["text1", "text2", "text3"]
        embeddings = np.random.rand(3, 384)
        
        # Store embeddings
        embedding_service.store_embeddings(text_ids, embeddings)
        
        # Retrieve embeddings
        retrieved = embedding_service.retrieve_embeddings(text_ids)
        
        assert len(retrieved) == 3
        assert all(id in retrieved for id in text_ids)
        assert all(retrieved[id] is not None for id in text_ids)

class TestSimilarityService:
    """Test cases for similarity service."""
    
    @pytest.fixture
    def similarity_service(self):
        """Create similarity service instance for testing."""
        return SimilarityService()
    
    @pytest.mark.asyncio
    async def test_semantic_similarity(self, similarity_service):
        """Test semantic similarity calculation."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        
        # Mock embedding service
        with patch('services.similarity_service.embedding_service') as mock_embed_service:
            mock_embed_service.generate_embeddings.return_value = np.array([
                [0.1, 0.2, 0.3, 0.4],  # Mock embedding for text1
                [0.15, 0.25, 0.35, 0.45]  # Similar mock embedding for text2
            ])
            
            result = await similarity_service.calculate_semantic_similarity(
                text1, text2, "text1", "text2"
            )
            
            assert result.text1_id == "text1"
            assert result.text2_id == "text2"
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.detection_method == "semantic_embedding"
    
    @pytest.mark.asyncio
    async def test_optimized_cosine_similarity(self, similarity_service):
        """Test optimized cosine similarity calculation."""
        embedding1 = np.array([1.0, 0.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0, 0.0])
        
        similarity = await similarity_service._calculate_cosine_similarity_optimized(
            embedding1, embedding2
        )
        
        assert similarity == 0.0  # Orthogonal vectors
        
        # Test identical vectors
        similarity = await similarity_service._calculate_cosine_similarity_optimized(
            embedding1, embedding1
        )
        
        assert similarity == 1.0  # Identical vectors
    
    @pytest.mark.asyncio
    async def test_enhanced_duplicate_detection(self, similarity_service):
        """Test enhanced duplicate detection with custom thresholds."""
        query_text = "This is the original text."
        candidates = {
            "candidate1": "This is the original text.",  # Exact duplicate
            "candidate2": "This is similar but different text.",
            "candidate3": "Completely unrelated content about space."
        }
        
        custom_thresholds = {
            "duplicate": 0.9,
            "potential": 0.6,
            "semantic": 0.85,
            "lexical": 0.8
        }
        
        # Mock embedding service
        with patch('services.similarity_service.embedding_service') as mock_embed_service:
            mock_embed_service.generate_embedding.side_effect = [
                np.array([1.0, 0.0, 0.0, 0.0]),  # Query
                np.array([1.0, 0.0, 0.0, 0.0]),  # Candidate1 (identical)
                np.array([0.8, 0.2, 0.0, 0.0]),  # Candidate2 (similar)
                np.array([0.0, 0.0, 1.0, 0.0])   # Candidate3 (different)
            ]
            
            result = await similarity_service.detect_duplicates(
                query_text, candidates, "query", 
                use_semantic=True, use_lexical=False,
                custom_thresholds=custom_thresholds
            )
            
            assert result.query_text_id == "query"
            assert len(result.duplicates) >= 1  # Should find at least the identical one
            assert result.duplicates[0].text2_id == "candidate1"
    
    @pytest.mark.asyncio
    async def test_batch_similarity_search(self, similarity_service):
        """Test efficient batch similarity search."""
        query_texts = ["Query text 1", "Query text 2"]
        candidate_texts = {
            "candidate1": "Similar to query 1",
            "candidate2": "Similar to query 2",
            "candidate3": "Different content"
        }
        
        # Mock embedding service
        with patch('services.similarity_service.embedding_service') as mock_embed_service:
            mock_embed_service.generate_batch_embeddings.return_value = [
                np.array([1.0, 0.0, 0.0, 0.0]),  # Query 1
                np.array([0.0, 1.0, 0.0, 0.0]),  # Query 2
                np.array([0.9, 0.1, 0.0, 0.0]),  # Candidate 1 (similar to query 1)
                np.array([0.1, 0.9, 0.0, 0.0]),  # Candidate 2 (similar to query 2)
                np.array([0.0, 0.0, 1.0, 0.0])   # Candidate 3 (different)
            ]
            
            results = await similarity_service.efficient_batch_similarity_search(
                query_texts, candidate_texts, batch_size=32
            )
            
            assert len(results) == 2  # Two queries
            assert "query_0" in results
            assert "query_1" in results
    
    @pytest.mark.asyncio
    async def test_enhanced_cross_student_plagiarism_detection(self, similarity_service):
        """Test enhanced cross-student plagiarism detection."""
        submissions = {
            "sub1": {"text": "Original student work", "student_id": "student1"},
            "sub2": {"text": "Original student work", "student_id": "student2"},  # Plagiarized
            "sub3": {"text": "Different original work", "student_id": "student3"}
        }
        
        # Mock the detect_duplicates method
        with patch.object(similarity_service, 'detect_duplicates') as mock_detect:
            mock_result = Mock()
            mock_result.duplicates = [
                Mock(
                    text1_id="sub1", text2_id="sub2", 
                    similarity_score=0.95, detection_method="semantic",
                    metadata={}
                )
            ]
            mock_result.potential_duplicates = []
            mock_detect.return_value = mock_result
            
            results = await similarity_service.cross_student_plagiarism_detection(
                submissions, plagiarism_threshold=0.8, batch_processing=False
            )
            
            # Should detect plagiarism for student1 (their work was copied by student2)
            assert "student1" in results
            assert len(results["student1"]) > 0
    
    def test_determine_duplicate_status(self, similarity_service):
        """Test duplicate status determination logic."""
        # Test with high similarity results
        high_sim_results = [
            Mock(is_duplicate=True, similarity_score=0.95),
            Mock(is_duplicate=False, similarity_score=0.7)
        ]
        thresholds = {"duplicate": 0.85}
        
        is_duplicate = similarity_service._determine_duplicate_status(high_sim_results, thresholds)
        assert is_duplicate is True
        
        # Test with low similarity results
        low_sim_results = [
            Mock(is_duplicate=False, similarity_score=0.6),
            Mock(is_duplicate=False, similarity_score=0.5)
        ]
        
        is_duplicate = similarity_service._determine_duplicate_status(low_sim_results, thresholds)
        assert is_duplicate is False
    
    def test_confidence_level_calculation(self, similarity_service):
        """Test confidence level calculation."""
        assert similarity_service._calculate_confidence_level([], 0.98) == "very_high"
        assert similarity_service._calculate_confidence_level([], 0.88) == "high"
        assert similarity_service._calculate_confidence_level([], 0.75) == "medium"
        assert similarity_service._calculate_confidence_level([], 0.55) == "low"
        assert similarity_service._calculate_confidence_level([], 0.3) == "very_low"
    
    def test_lexical_similarity(self, similarity_service):
        """Test lexical similarity calculation."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A quick brown fox leaps over a lazy dog."
        
        result = similarity_service.calculate_lexical_similarity(
            text1, text2, "text1", "text2"
        )
        
        assert result.text1_id == "text1"
        assert result.text2_id == "text2"
        assert 0.0 <= result.similarity_score <= 1.0
        assert result.detection_method == "lexical_tfidf"
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, similarity_service):
        """Test duplicate detection functionality."""
        query_text = "This is the original text."
        candidates = {
            "candidate1": "This is the original text.",  # Exact duplicate
            "candidate2": "This is similar but different text.",
            "candidate3": "Completely unrelated content about space."
        }
        
        # Mock semantic similarity
        with patch.object(similarity_service, 'calculate_semantic_similarity') as mock_semantic:
            mock_semantic.side_effect = [
                Mock(similarity_score=1.0, is_duplicate=True, detection_method="semantic"),
                Mock(similarity_score=0.6, is_duplicate=False, detection_method="semantic"),
                Mock(similarity_score=0.2, is_duplicate=False, detection_method="semantic")
            ]
            
            # Mock lexical similarity
            with patch.object(similarity_service, 'calculate_lexical_similarity') as mock_lexical:
                mock_lexical.side_effect = [
                    Mock(similarity_score=1.0, is_duplicate=True, detection_method="lexical"),
                    Mock(similarity_score=0.5, is_duplicate=False, detection_method="lexical"),
                    Mock(similarity_score=0.1, is_duplicate=False, detection_method="lexical")
                ]
                
                result = await similarity_service.detect_duplicates(
                    query_text, candidates, "query"
                )
                
                assert result.query_text_id == "query"
                assert len(result.duplicates) == 1  # Only candidate1 should be duplicate
                assert result.duplicates[0].text2_id == "candidate1"

class TestVectorDatabase:
    """Test cases for vector database."""
    
    @pytest.fixture
    def vector_db(self):
        """Create vector database instance for testing."""
        return VectorDatabase(storage_path="./test_vector_db")
    
    def test_add_vector(self, vector_db):
        """Test adding vectors to database."""
        vector = np.random.rand(384)
        metadata = {"text": "test text", "student_id": "student1"}
        
        success = vector_db.add_vector("test_id", vector, metadata)
        
        assert success is True
        assert "test_id" in vector_db.records
        
        # Test duplicate ID
        success = vector_db.add_vector("test_id", vector, metadata)
        assert success is False
    
    def test_batch_add_vectors(self, vector_db):
        """Test batch adding vectors."""
        vectors_data = [
            ("id1", np.random.rand(384), {"text": "text1"}),
            ("id2", np.random.rand(384), {"text": "text2"}),
            ("id3", np.random.rand(384), {"text": "text3"})
        ]
        
        added_count = vector_db.add_vectors_batch(vectors_data)
        
        assert added_count == 3
        assert len(vector_db.records) == 3
    
    def test_build_index(self, vector_db):
        """Test building search index."""
        # Add some vectors first
        for i in range(5):
            vector = np.random.rand(384)
            vector_db.add_vector(f"id{i}", vector, {"text": f"text{i}"})
        
        vector_db.build_index()
        
        assert vector_db.is_indexed is True
        assert vector_db.index is not None
        assert len(vector_db.vector_ids) == 5
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_db):
        """Test similarity search functionality."""
        # Add some vectors
        vectors = [np.random.rand(384) for _ in range(3)]
        for i, vector in enumerate(vectors):
            vector_db.add_vector(f"id{i}", vector, {"text": f"text{i}", "student_id": f"student{i}"})
        
        vector_db.build_index()
        
        # Search with first vector (should find itself)
        results = await vector_db.search_similar(vectors[0], k=2, threshold=0.5)
        
        assert len(results) >= 1
        assert results[0][0] == "id0"  # Should find itself first
        assert results[0][1] >= 0.99  # High similarity with itself
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self, vector_db):
        """Test similarity search with metadata filtering."""
        # Add vectors with different student IDs
        vectors = [np.random.rand(384) for _ in range(4)]
        for i, vector in enumerate(vectors):
            vector_db.add_vector(f"id{i}", vector, {"student_id": f"student{i % 2}"})
        
        vector_db.build_index()
        
        # Search excluding student0
        filter_metadata = {"student_id": {"not": "student0"}}
        results = await vector_db.search_similar(
            vectors[0], k=5, threshold=0.0, filter_metadata=filter_metadata
        )
        
        # Should not include vectors from student0
        for vector_id, similarity, metadata in results:
            assert metadata.get("student_id") != "student0"
    
    @pytest.mark.asyncio
    async def test_find_duplicates_with_exclusion(self, vector_db):
        """Test finding duplicates with same-student exclusion."""
        # Add similar vectors from different students
        base_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        similar_vector = np.array([0.95, 0.05, 0.0] + [0.0] * 381)
        
        vector_db.add_vector("sub1", base_vector, {"student_id": "student1"})
        vector_db.add_vector("sub2", similar_vector, {"student_id": "student2"})
        vector_db.add_vector("sub3", base_vector * 0.9, {"student_id": "student1"})  # Same student
        
        vector_db.build_index()
        
        # Find duplicates excluding same student
        duplicates = await vector_db.find_duplicates(
            similarity_threshold=0.8,
            exclude_same_metadata={"student_id": "same"}
        )
        
        # Should find cross-student duplicates but not same-student ones
        cross_student_pairs = [
            (id1, id2) for id1, id2, sim in duplicates
            if vector_db.records[id1].metadata["student_id"] != vector_db.records[id2].metadata["student_id"]
        ]
        
        assert len(cross_student_pairs) > 0
    
    @pytest.mark.asyncio
    async def test_batch_similarity_search(self, vector_db):
        """Test batch similarity search."""
        # Add some vectors
        vectors = [np.random.rand(384) for _ in range(5)]
        for i, vector in enumerate(vectors):
            vector_db.add_vector(f"id{i}", vector, {"text": f"text{i}"})
        
        vector_db.build_index()
        
        # Perform batch search
        query_vectors = [vectors[0], vectors[1]]
        results = await vector_db.batch_similarity_search(query_vectors, k=3, threshold=0.5)
        
        assert len(results) == 2  # Two query results
        assert all(len(result) >= 1 for result in results)  # Each should find at least itself

class TestDuplicateDetectionService:
    """Test cases for duplicate detection service."""
    
    @pytest.fixture
    def detection_service(self):
        """Create duplicate detection service instance for testing."""
        return DuplicateDetectionService()
    
    def test_detection_level_setting(self, detection_service):
        """Test setting detection levels."""
        detection_service.set_detection_level(DetectionLevel.STRICT)
        assert detection_service.current_level == DetectionLevel.STRICT
        
        detection_service.set_detection_level(DetectionLevel.SENSITIVE)
        assert detection_service.current_level == DetectionLevel.SENSITIVE
    
    @pytest.mark.asyncio
    async def test_submission_analysis(self, detection_service):
        """Test submission analysis functionality."""
        submission_text = "This is a test submission."
        existing_submissions = {
            "sub1": {"text": "This is a test submission.", "student_id": "student2"},
            "sub2": {"text": "Different content entirely.", "student_id": "student3"}
        }
        
        # Mock the similarity service
        with patch('services.duplicate_detection_service.similarity_service') as mock_sim:
            mock_result = Mock()
            mock_result.duplicates = [
                Mock(text2_id="sub1", similarity_score=0.95, detection_method="semantic", metadata={})
            ]
            mock_result.potential_duplicates = []
            mock_result.total_comparisons = 2
            mock_result.processing_time = 0.1
            
            mock_sim.detect_duplicates.return_value = mock_result
            
            report = await detection_service.analyze_submission(
                submission_id="test_sub",
                submission_text=submission_text,
                student_id="student1",
                existing_submissions=existing_submissions
            )
            
            assert report.query_submission_id == "test_sub"
            assert report.duplicates_found == 1
            assert len(report.alerts) >= 1
    
    def test_statistics(self, detection_service):
        """Test getting detection statistics."""
        stats = detection_service.get_detection_statistics()
        
        assert "detection_level" in stats
        assert "thresholds" in stats
        assert "vector_database" in stats
        assert "embedding_model" in stats

if __name__ == "__main__":
    pytest.main([__file__])
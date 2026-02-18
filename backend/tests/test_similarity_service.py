"""
Tests for similarity service functionality.
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from services.similarity_service import SimilarityService, MatchType, SimilarityMatch
from services.embedding_service import EmbeddingService
from utils.exceptions import SimilarityError, DuplicateDetectionError

class TestSimilarityService:
    """Test cases for SimilarityService."""
    
    @pytest.fixture
    def similarity_service(self):
        """Create a SimilarityService instance for testing."""
        return SimilarityService()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return {
            'identical': np.array([1.0, 0.0, 0.0, 0.0]),
            'similar': np.array([0.9, 0.1, 0.0, 0.0]),
            'different': np.array([0.0, 0.0, 1.0, 0.0])
        }
    
    @pytest.mark.asyncio
    async def test_calculate_cosine_similarity_identical(self, similarity_service, sample_embeddings):
        """Test cosine similarity calculation for identical embeddings."""
        embedding = sample_embeddings['identical']
        similarity = await similarity_service.calculate_cosine_similarity(embedding, embedding)
        
        assert similarity == 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_cosine_similarity_different(self, similarity_service, sample_embeddings):
        """Test cosine similarity calculation for different embeddings."""
        embedding1 = sample_embeddings['identical']
        embedding2 = sample_embeddings['different']
        similarity = await similarity_service.calculate_cosine_similarity(embedding1, embedding2)
        
        assert similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_cosine_similarity_similar(self, similarity_service, sample_embeddings):
        """Test cosine similarity calculation for similar embeddings."""
        embedding1 = sample_embeddings['identical']
        embedding2 = sample_embeddings['similar']
        similarity = await similarity_service.calculate_cosine_similarity(embedding1, embedding2)
        
        assert 0.8 < similarity < 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_cosine_similarity_zero_vector(self, similarity_service):
        """Test cosine similarity with zero vectors."""
        zero_vector = np.array([0.0, 0.0, 0.0, 0.0])
        normal_vector = np.array([1.0, 0.0, 0.0, 0.0])
        
        similarity = await similarity_service.calculate_cosine_similarity(zero_vector, normal_vector)
        assert similarity == 0.0
    
    def test_determine_match_type(self, similarity_service):
        """Test match type determination based on similarity scores."""
        # Test exact match
        assert similarity_service._determine_match_type(1.0) == MatchType.EXACT
        
        # Test near duplicate
        assert similarity_service._determine_match_type(0.96) == MatchType.NEAR_DUPLICATE
        
        # Test semantic similarity
        assert similarity_service._determine_match_type(0.87) == MatchType.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_find_similar_submissions_empty_index(self, similarity_service):
        """Test finding similar submissions with empty vector index."""
        results = await similarity_service.find_similar_submissions("test content")
        assert results == []
    
    @pytest.mark.asyncio
    @patch('services.similarity_service.embedding_service.generate_embedding')
    async def test_add_to_vector_index(self, mock_generate_embedding, similarity_service):
        """Test adding submission to vector index."""
        mock_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        mock_generate_embedding.return_value = mock_embedding
        
        await similarity_service._add_to_vector_index("sub_1", "test content", "student_1")
        
        assert "sub_1" in similarity_service.vector_index
        assert "sub_1" in similarity_service.submission_metadata
        assert similarity_service.vector_index["sub_1"]["student_id"] == "student_1"
    
    @pytest.mark.asyncio
    async def test_get_similarity_statistics(self, similarity_service):
        """Test getting similarity statistics."""
        stats = await similarity_service.get_similarity_statistics()
        
        assert "indexed_submissions" in stats
        assert "cached_similarities" in stats
        assert "thresholds" in stats
        assert "vector_index_size_mb" in stats
    
    @pytest.mark.asyncio
    async def test_update_thresholds(self, similarity_service):
        """Test updating similarity thresholds."""
        original_threshold = similarity_service.review_threshold
        new_threshold = 0.75
        
        await similarity_service.update_thresholds(review_threshold=new_threshold)
        
        assert similarity_service.review_threshold == new_threshold
        assert similarity_service.review_threshold != original_threshold
    
    @pytest.mark.asyncio
    async def test_clear_vector_index(self, similarity_service):
        """Test clearing vector index."""
        # Add some data first
        similarity_service.vector_index["test"] = {"data": "test"}
        similarity_service.submission_metadata["test"] = {"meta": "test"}
        similarity_service.similarity_cache["test"] = 0.5
        
        await similarity_service.clear_vector_index()
        
        assert len(similarity_service.vector_index) == 0
        assert len(similarity_service.submission_metadata) == 0
        assert len(similarity_service.similarity_cache) == 0

class TestEmbeddingService:
    """Test cases for EmbeddingService."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance for testing."""
        return EmbeddingService()
    
    def test_get_cache_key(self, embedding_service):
        """Test cache key generation."""
        text = "test content"
        model_name = "sentence-bert"
        
        key1 = embedding_service._get_cache_key(text, model_name)
        key2 = embedding_service._get_cache_key(text, model_name)
        
        # Same input should generate same key
        assert key1 == key2
        
        # Different text should generate different key
        key3 = embedding_service._get_cache_key("different text", model_name)
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_get_embedding_info(self, embedding_service):
        """Test getting embedding model information."""
        info = await embedding_service.get_embedding_info("sentence-bert")
        
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "max_sequence_length" in info
        assert "is_loaded" in info
        assert "device" in info
    
    def test_get_cache_stats(self, embedding_service):
        """Test getting cache statistics."""
        stats = embedding_service.get_cache_stats()
        
        assert "cached_embeddings" in stats
        assert "total_cache_size_mb" in stats
        assert "cache_directory" in stats
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, embedding_service):
        """Test clearing embedding cache."""
        # This should not raise an exception even if cache is empty
        await embedding_service.clear_cache()
        
        # Verify cache stats show empty cache
        stats = embedding_service.get_cache_stats()
        assert stats["cached_embeddings"] == 0
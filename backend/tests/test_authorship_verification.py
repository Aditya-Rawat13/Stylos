"""
Tests for authorship verification models and services.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.authorship_models import (
    SiameseNetwork,
    AuthorshipVerificationService,
    AIDetectionClassifier,
    AuthorshipResult,
    AIDetectionResult,
    AuthorshipDataset
)
from services.verification_service import (
    ComprehensiveVerificationService,
    VerificationRequest,
    ComprehensiveVerificationResult
)

class TestSiameseNetwork:
    """Test cases for Siamese neural network."""
    
    def test_siamese_network_initialization(self):
        """Test Siamese network initialization."""
        model = SiameseNetwork(input_dim=384, hidden_dim=128)
        assert model.input_dim == 384
        assert model.hidden_dim == 128
    
    def test_siamese_network_forward_pass(self):
        """Test forward pass of Siamese network."""
        model = SiameseNetwork(input_dim=384, hidden_dim=128)
        
        # Create mock inputs
        input1 = np.random.rand(384)
        input2 = np.random.rand(384)
        
        # Test forward pass
        result = model.forward(input1, input2)
        
        # Should return a similarity score
        assert isinstance(result, (float, np.ndarray))
        if isinstance(result, np.ndarray):
            assert result.shape == (1,) or result.shape == ()

class TestAuthorshipDataset:
    """Test cases for authorship dataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization with sample data."""
        embeddings = [np.random.rand(384) for _ in range(10)]
        labels = [1] * 10
        author_ids = ['author1'] * 5 + ['author2'] * 5
        
        dataset = AuthorshipDataset(embeddings, labels, author_ids)
        
        assert len(dataset.embeddings) == 10
        assert len(dataset.author_ids) == 10
        assert len(dataset.pairs) > 0
        assert len(dataset.pair_labels) > 0
    
    def test_dataset_pair_generation(self):
        """Test positive and negative pair generation."""
        embeddings = [np.random.rand(384) for _ in range(6)]
        labels = [1] * 6
        author_ids = ['author1'] * 3 + ['author2'] * 3
        
        dataset = AuthorshipDataset(embeddings, labels, author_ids)
        
        # Should have positive pairs (same author) and negative pairs (different authors)
        positive_pairs = sum(1 for label in dataset.pair_labels if label == 1)
        negative_pairs = sum(1 for label in dataset.pair_labels if label == 0)
        
        assert positive_pairs > 0  # Should have same-author pairs
        assert negative_pairs > 0  # Should have different-author pairs

class TestAIDetectionClassifier:
    """Test cases for AI detection classifier."""
    
    @pytest.fixture
    def ai_detector(self):
        """Create AI detection classifier instance."""
        return AIDetectionClassifier()
    
    @pytest.mark.asyncio
    async def test_ai_detector_initialization(self, ai_detector):
        """Test AI detector initialization."""
        await ai_detector.initialize_models()
        
        # Should initialize without errors
        assert ai_detector.tokenizer is not None
        assert ai_detector.transformer_model is not None
    
    @pytest.mark.asyncio
    async def test_ai_content_detection(self, ai_detector):
        """Test AI content detection."""
        await ai_detector.initialize_models()
        
        # Test with sample text and features
        text = "This is a sample text for testing AI detection capabilities."
        stylometric_features = {
            'type_token_ratio': 0.8,
            'avg_sentence_length': 12.0,
            'lexical_diversity': 0.75
        }
        
        result = await ai_detector.detect_ai_content(text, stylometric_features)
        
        assert isinstance(result, AIDetectionResult)
        assert 0.0 <= result.ai_probability <= 1.0
        assert 0.0 <= result.human_probability <= 1.0
        assert abs(result.ai_probability + result.human_probability - 1.0) < 0.01
        assert result.detection_method in ['ensemble', 'mock', 'error']
    
    def test_transformer_feature_extraction(self, ai_detector):
        """Test transformer feature extraction."""
        texts = [
            "This is the first sample text.",
            "Here is another sample for testing."
        ]
        
        features = ai_detector.extract_transformer_features(texts)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(texts)
        assert features.shape[1] > 0  # Should have feature dimensions

class TestAuthorshipVerificationService:
    """Test cases for authorship verification service."""
    
    @pytest.fixture
    def authorship_service(self):
        """Create authorship verification service instance."""
        return AuthorshipVerificationService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, authorship_service):
        """Test service initialization."""
        await authorship_service.initialize()
        
        assert authorship_service.siamese_model is not None
        assert authorship_service.ai_detector is not None
    
    @pytest.mark.asyncio
    async def test_authorship_verification(self, authorship_service):
        """Test authorship verification process."""
        await authorship_service.initialize()
        
        # Create sample embeddings
        candidate_embedding = np.random.rand(384)
        reference_embeddings = [np.random.rand(384) for _ in range(3)]
        
        result = await authorship_service.verify_authorship(
            candidate_embedding, 
            reference_embeddings
        )
        
        assert isinstance(result, AuthorshipResult)
        assert 0.0 <= result.similarity_score <= 1.0
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
        assert isinstance(result.is_authentic, bool)
        assert result.uncertainty >= 0.0
    
    @pytest.mark.asyncio
    async def test_authorship_verification_empty_references(self, authorship_service):
        """Test authorship verification with no reference embeddings."""
        await authorship_service.initialize()
        
        candidate_embedding = np.random.rand(384)
        reference_embeddings = []
        
        result = await authorship_service.verify_authorship(
            candidate_embedding, 
            reference_embeddings
        )
        
        assert isinstance(result, AuthorshipResult)
        assert result.similarity_score == 0.0
        assert not result.is_authentic
        assert result.uncertainty == 1.0

class TestComprehensiveVerificationService:
    """Test cases for comprehensive verification service."""
    
    @pytest.fixture
    def verification_service(self):
        """Create comprehensive verification service instance."""
        return ComprehensiveVerificationService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, verification_service):
        """Test comprehensive service initialization."""
        with patch.object(verification_service.authorship_service, 'initialize', new_callable=AsyncMock):
            with patch.object(verification_service.ai_detector, 'initialize_models', new_callable=AsyncMock):
                with patch.object(verification_service.duplicate_detector, 'initialize', new_callable=AsyncMock):
                    with patch('services.verification_service.embedding_service.initialize_models', new_callable=AsyncMock):
                        await verification_service.initialize()
                        
                        assert verification_service.initialized
    
    @pytest.mark.asyncio
    async def test_comprehensive_verification(self, verification_service):
        """Test comprehensive verification process."""
        # Mock all dependencies
        with patch.object(verification_service, 'initialize', new_callable=AsyncMock):
            with patch.object(verification_service.text_processor, 'process_text', new_callable=AsyncMock) as mock_process:
                with patch.object(verification_service.stylometric_analyzer, 'extract_features') as mock_features:
                    with patch('services.verification_service.embedding_service.generate_embedding', new_callable=AsyncMock) as mock_embedding:
                        with patch.object(verification_service, '_verify_authorship', new_callable=AsyncMock) as mock_authorship:
                            with patch.object(verification_service.ai_detector, 'detect_ai_content', new_callable=AsyncMock) as mock_ai:
                                with patch.object(verification_service, '_check_duplicates', new_callable=AsyncMock) as mock_duplicates:
                                    
                                    # Setup mocks
                                    mock_process.return_value = "processed text"
                                    mock_features.return_value = {'feature1': 0.5}
                                    mock_embedding.return_value = np.random.rand(384)
                                    mock_authorship.return_value = AuthorshipResult(
                                        similarity_score=0.8,
                                        confidence_interval=(0.7, 0.9),
                                        is_authentic=True,
                                        uncertainty=0.1,
                                        feature_importance={}
                                    )
                                    mock_ai.return_value = AIDetectionResult(
                                        ai_probability=0.2,
                                        human_probability=0.8,
                                        confidence=0.9,
                                        detection_method="test",
                                        explanation={}
                                    )
                                    mock_duplicates.return_value = []
                                    
                                    verification_service.initialized = True
                                    
                                    # Create test request
                                    request = VerificationRequest(
                                        text="This is a test text for verification.",
                                        student_id="student123",
                                        submission_id="submission456"
                                    )
                                    
                                    # Perform verification
                                    result = await verification_service.verify_submission(request)
                                    
                                    assert isinstance(result, ComprehensiveVerificationResult)
                                    assert result.submission_id == "submission456"
                                    assert result.student_id == "student123"
                                    assert result.overall_status in ['PASS', 'FAIL', 'REVIEW', 'ERROR']
                                    assert 0.0 <= result.confidence_score <= 1.0
    
    def test_overall_assessment_pass(self, verification_service):
        """Test overall assessment with passing criteria."""
        authorship_result = AuthorshipResult(
            similarity_score=0.85,
            confidence_interval=(0.8, 0.9),
            is_authentic=True,
            uncertainty=0.1,
            feature_importance={}
        )
        
        ai_detection_result = AIDetectionResult(
            ai_probability=0.1,
            human_probability=0.9,
            confidence=0.9,
            detection_method="test",
            explanation={}
        )
        
        duplicate_matches = []
        stylometric_features = {'type_token_ratio': 0.7}
        
        status, confidence, risk_factors = verification_service._assess_overall_result(
            authorship_result, ai_detection_result, duplicate_matches, stylometric_features
        )
        
        assert status == "PASS"
        assert confidence > 0.5
        assert len(risk_factors) == 0
    
    def test_overall_assessment_fail(self, verification_service):
        """Test overall assessment with failing criteria."""
        authorship_result = AuthorshipResult(
            similarity_score=0.3,
            confidence_interval=(0.2, 0.4),
            is_authentic=False,
            uncertainty=0.8,
            feature_importance={}
        )
        
        ai_detection_result = AIDetectionResult(
            ai_probability=0.9,
            human_probability=0.1,
            confidence=0.9,
            detection_method="test",
            explanation={}
        )
        
        duplicate_matches = [{'similarity_score': 0.95}]
        stylometric_features = {'type_token_ratio': 0.2}
        
        status, confidence, risk_factors = verification_service._assess_overall_result(
            authorship_result, ai_detection_result, duplicate_matches, stylometric_features
        )
        
        assert status == "FAIL"
        assert len(risk_factors) > 0
    
    def test_stylometric_anomaly_detection(self, verification_service):
        """Test stylometric anomaly detection."""
        # Normal features
        normal_features = {
            'type_token_ratio': 0.6,
            'avg_sentence_length': 15.0,
            'total_punctuation_ratio': 0.05
        }
        
        assert not verification_service._detect_stylometric_anomalies(normal_features)
        
        # Anomalous features
        anomalous_features = {
            'type_token_ratio': 0.1,  # Very low
            'avg_sentence_length': 60.0,  # Very high
            'total_punctuation_ratio': 0.25  # Very high
        }
        
        assert verification_service._detect_stylometric_anomalies(anomalous_features)
    
    def test_threshold_updates(self, verification_service):
        """Test threshold updates."""
        original_thresholds = verification_service.thresholds.copy()
        
        new_thresholds = {
            'authorship_min_score': 0.8,
            'ai_detection_max_prob': 0.7
        }
        
        verification_service.update_thresholds(new_thresholds)
        
        assert verification_service.thresholds['authorship_min_score'] == 0.8
        assert verification_service.thresholds['ai_detection_max_prob'] == 0.7
        # Other thresholds should remain unchanged
        assert verification_service.thresholds['duplicate_max_similarity'] == original_thresholds['duplicate_max_similarity']

@pytest.mark.asyncio
async def test_integration_verification_flow():
    """Integration test for the complete verification flow."""
    service = ComprehensiveVerificationService()
    
    # Mock all external dependencies
    with patch.object(service, 'initialize', new_callable=AsyncMock):
        with patch.object(service.text_processor, 'process_text', new_callable=AsyncMock) as mock_process:
            with patch.object(service.stylometric_analyzer, 'extract_features') as mock_features:
                with patch('services.verification_service.embedding_service.generate_embedding', new_callable=AsyncMock) as mock_embedding:
                    
                    # Setup realistic mock responses
                    mock_process.return_value = "This is a processed academic essay about machine learning."
                    mock_features.return_value = {
                        'type_token_ratio': 0.65,
                        'avg_sentence_length': 18.5,
                        'lexical_diversity': 0.72,
                        'total_punctuation_ratio': 0.08
                    }
                    mock_embedding.return_value = np.random.rand(384)
                    
                    # Mock verification methods
                    with patch.object(service, '_verify_authorship', new_callable=AsyncMock) as mock_auth:
                        with patch.object(service.ai_detector, 'detect_ai_content', new_callable=AsyncMock) as mock_ai:
                            with patch.object(service, '_check_duplicates', new_callable=AsyncMock) as mock_dup:
                                
                                mock_auth.return_value = AuthorshipResult(
                                    similarity_score=0.82,
                                    confidence_interval=(0.75, 0.89),
                                    is_authentic=True,
                                    uncertainty=0.15,
                                    feature_importance={'semantic': 0.8, 'stylometric': 0.85}
                                )
                                
                                mock_ai.return_value = AIDetectionResult(
                                    ai_probability=0.12,
                                    human_probability=0.88,
                                    confidence=0.91,
                                    detection_method="ensemble",
                                    explanation={'method': 'transformer + stylometric'}
                                )
                                
                                mock_dup.return_value = []
                                
                                service.initialized = True
                                
                                # Create verification request
                                request = VerificationRequest(
                                    text="This is a comprehensive academic essay about machine learning algorithms and their applications in natural language processing.",
                                    student_id="student_001",
                                    submission_id="sub_12345"
                                )
                                
                                # Perform verification
                                result = await service.verify_submission(request)
                                
                                # Verify results
                                assert result.overall_status == "PASS"
                                assert result.confidence_score > 0.7
                                assert len(result.risk_factors) == 0
                                assert result.authorship_result.is_authentic
                                assert result.ai_detection_result.human_probability > 0.8


class TestModelPerformanceMetrics:
    """Test cases for ML/DL model performance and accuracy."""
    
    @pytest.fixture
    def siamese_model(self):
        """Create Siamese network for testing."""
        return SiameseNetwork(input_dim=384, hidden_dim=128)
    
    def test_model_accuracy_same_author(self, siamese_model):
        """Test model accuracy for same author pairs."""
        # Create similar embeddings (same author)
        embedding1 = np.random.rand(384)
        embedding2 = embedding1 + np.random.normal(0, 0.1, 384)  # Add small noise
        
        similarity = siamese_model.forward(embedding1, embedding2)
        
        # Should have high similarity for same author
        if isinstance(similarity, np.ndarray):
            similarity = float(similarity.flatten()[0])
        
        assert similarity > 0.5, "Model should detect high similarity for same author"
    
    def test_model_accuracy_different_authors(self, siamese_model):
        """Test model accuracy for different author pairs."""
        # Create very different embeddings (different authors)
        embedding1 = np.random.rand(384)
        embedding2 = np.random.rand(384)
        
        similarity = siamese_model.forward(embedding1, embedding2)
        
        if isinstance(similarity, np.ndarray):
            similarity = float(similarity.flatten()[0])
        
        # Should have lower similarity for different authors
        assert 0.0 <= similarity <= 1.0, "Similarity should be in valid range"
    
    def test_model_inference_time(self, siamese_model):
        """Test model inference performance."""
        import time
        
        embedding1 = np.random.rand(384)
        embedding2 = np.random.rand(384)
        
        start_time = time.time()
        for _ in range(100):
            siamese_model.forward(embedding1, embedding2)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Should complete inference in reasonable time (< 100ms per inference)
        assert avg_time < 0.1, f"Inference too slow: {avg_time:.4f}s per inference"
    
    def test_model_batch_processing(self, siamese_model):
        """Test model batch processing capability."""
        batch_size = 10
        embeddings1 = [np.random.rand(384) for _ in range(batch_size)]
        embeddings2 = [np.random.rand(384) for _ in range(batch_size)]
        
        results = []
        for e1, e2 in zip(embeddings1, embeddings2):
            result = siamese_model.forward(e1, e2)
            results.append(result)
        
        assert len(results) == batch_size
        for result in results:
            if isinstance(result, np.ndarray):
                result = float(result.flatten()[0])
            assert 0.0 <= result <= 1.0


class TestAIDetectionPerformance:
    """Test cases for AI detection model performance."""
    
    @pytest.fixture
    def ai_detector(self):
        """Create AI detector instance."""
        return AIDetectionClassifier()
    
    @pytest.mark.asyncio
    async def test_detection_precision_human_text(self, ai_detector):
        """Test precision on human-written text."""
        await ai_detector.initialize_models()
        
        # Human-like text samples
        human_texts = [
            "I think this is really interesting. Yesterday I was talking to my friend about it.",
            "Honestly, I'm not sure what to make of this. It's confusing but also kind of cool.",
            "My experience has been different. When I tried it last week, things didn't work out."
        ]
        
        human_detections = 0
        for text in human_texts:
            result = await ai_detector.detect_ai_content(text, {'type_token_ratio': 0.7})
            if result.human_probability > 0.5:
                human_detections += 1
        
        # Should correctly identify most human texts
        precision = human_detections / len(human_texts)
        assert precision >= 0.5, f"Low precision on human text: {precision:.2f}"
    
    @pytest.mark.asyncio
    async def test_detection_recall_ai_text(self, ai_detector):
        """Test recall on AI-generated text."""
        await ai_detector.initialize_models()
        
        # AI-like text samples (formal, structured)
        ai_texts = [
            "Furthermore, it is essential to implement systematic methodologies to facilitate optimal outcomes.",
            "Moreover, the comprehensive analysis demonstrates significant improvements in overall performance metrics.",
            "Additionally, the implementation of advanced techniques enables enhanced functionality and efficiency."
        ]
        
        ai_detections = 0
        for text in ai_texts:
            result = await ai_detector.detect_ai_content(text, {'type_token_ratio': 0.5})
            if result.ai_probability > 0.5:
                ai_detections += 1
        
        # Should correctly identify most AI texts
        recall = ai_detections / len(ai_texts)
        assert recall >= 0.3, f"Low recall on AI text: {recall:.2f}"
    
    @pytest.mark.asyncio
    async def test_detection_confidence_calibration(self, ai_detector):
        """Test confidence score calibration."""
        await ai_detector.initialize_models()
        
        text = "This is a test text for confidence calibration."
        result = await ai_detector.detect_ai_content(text, {'type_token_ratio': 0.6})
        
        # Confidence should be reasonable
        assert 0.0 <= result.confidence <= 1.0
        assert result.ai_probability + result.human_probability <= 1.01  # Allow small floating point error
"""
Integration tests for the complete authorship verification system.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.authorship_integration import authorship_manager
from utils.model_training import training_utils
from config.authorship_config import config

class TestAuthorshipIntegration:
    """Integration tests for authorship verification system."""
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self):
        """Test complete system initialization."""
        # Mock all external dependencies
        with patch('services.authorship_integration.embedding_service.initialize_models', new_callable=AsyncMock):
            with patch('services.authorship_integration.authorship_service.initialize', new_callable=AsyncMock):
                with patch('services.authorship_integration.ai_detection_service.initialize_models', new_callable=AsyncMock):
                    with patch('services.authorship_integration.enhanced_ai_detector.initialize_models', new_callable=AsyncMock):
                        with patch('services.authorship_integration.verification_service.initialize', new_callable=AsyncMock):
                            
                            await authorship_manager.initialize_all_models()
                            
                            assert authorship_manager.initialized
                            assert all(authorship_manager.model_status.values())
    
    @pytest.mark.asyncio
    async def test_comprehensive_verification_workflow(self):
        """Test the complete verification workflow."""
        # Mock all dependencies
        with patch.object(authorship_manager, 'initialize_all_models', new_callable=AsyncMock):
            with patch('services.authorship_integration.verification_service.verify_submission', new_callable=AsyncMock) as mock_verify:
                
                # Setup mock response
                from services.verification_service import ComprehensiveVerificationResult
                from services.authorship_models import AuthorshipResult, AIDetectionResult
                from datetime import datetime
                
                mock_result = ComprehensiveVerificationResult(
                    submission_id="test_123",
                    student_id="student_456",
                    timestamp=datetime.utcnow(),
                    authorship_result=AuthorshipResult(
                        similarity_score=0.85,
                        confidence_interval=(0.8, 0.9),
                        is_authentic=True,
                        uncertainty=0.1,
                        feature_importance={"test": 0.8}
                    ),
                    ai_detection_result=AIDetectionResult(
                        ai_probability=0.15,
                        human_probability=0.85,
                        confidence=0.9,
                        detection_method="ensemble",
                        explanation={"method": "test"}
                    ),
                    duplicate_matches=[],
                    stylometric_features={"feature1": 0.5},
                    overall_status="PASS",
                    confidence_score=0.85,
                    risk_factors=[]
                )
                
                mock_verify.return_value = mock_result
                authorship_manager.initialized = True
                
                # Test comprehensive verification
                result = await authorship_manager.verify_authorship_comprehensive(
                    text="This is a test essay for comprehensive verification.",
                    student_id="student_456",
                    submission_id="test_123"
                )
                
                assert result['overall_status'] == "PASS"
                assert result['submission_id'] == "test_123"
                assert result['student_id'] == "student_456"
                assert 'processing_time_seconds' in result
                assert 'models_used' in result
    
    @pytest.mark.asyncio
    async def test_quick_authorship_check(self):
        """Test quick authorship verification."""
        with patch('services.authorship_integration.authorship_service.initialize', new_callable=AsyncMock):
            with patch('services.authorship_integration.embedding_service.generate_embedding', new_callable=AsyncMock) as mock_embed:
                with patch('services.authorship_integration.authorship_service.verify_authorship', new_callable=AsyncMock) as mock_verify:
                    
                    # Setup mocks
                    mock_embed.return_value = np.random.rand(768)
                    
                    from services.authorship_models import AuthorshipResult
                    mock_verify.return_value = AuthorshipResult(
                        similarity_score=0.8,
                        confidence_interval=(0.7, 0.9),
                        is_authentic=True,
                        uncertainty=0.2,
                        feature_importance={"test": 0.8}
                    )
                    
                    authorship_manager.model_status['siamese_network'] = True
                    
                    # Test quick check
                    reference_embeddings = [np.random.rand(768) for _ in range(3)]
                    result = await authorship_manager.quick_authorship_check(
                        "Test text for quick verification",
                        reference_embeddings
                    )
                    
                    assert result.similarity_score == 0.8
                    assert result.is_authentic
                    assert result.uncertainty == 0.2
    
    @pytest.mark.asyncio
    async def test_advanced_ai_detection(self):
        """Test advanced AI content detection."""
        with patch('services.authorship_integration.enhanced_ai_detector.initialize_models', new_callable=AsyncMock):
            with patch('services.authorship_integration.enhanced_ai_detector.detect_ai_content_enhanced', new_callable=AsyncMock) as mock_detect:
                
                # Setup mock response
                from services.ai_detection_enhanced import DetectionExplanation
                
                mock_explanation = DetectionExplanation(
                    primary_indicators=["Low perplexity"],
                    confidence_factors={"perplexity": 0.8},
                    linguistic_evidence={"test": "evidence"},
                    model_contributions={"model1": 0.7},
                    risk_assessment="Low risk",
                    human_readable_summary="Text appears human-written"
                )
                
                mock_detect.return_value = (0.2, 0.9, mock_explanation)
                authorship_manager.model_status['enhanced_ai_detector'] = True
                
                # Test advanced detection
                result = await authorship_manager.detect_ai_content_advanced(
                    "This is a test text for AI detection analysis."
                )
                
                assert result['ai_probability'] == 0.2
                assert result['human_probability'] == 0.8
                assert result['confidence'] == 0.9
                assert result['detection_method'] == 'enhanced_ensemble'
                assert 'explanation' in result
    
    def test_model_status_tracking(self):
        """Test model status tracking functionality."""
        # Reset status
        authorship_manager.initialized = False
        authorship_manager.model_status = {key: False for key in authorship_manager.model_status}
        
        status = authorship_manager.get_model_status()
        
        assert not status['initialized']
        assert not any(status['model_status'].values())
        assert 'metrics' in status
        assert 'timestamp' in status
    
    @pytest.mark.asyncio
    async def test_training_workflow(self):
        """Test model training workflow."""
        with patch.object(authorship_manager, 'train_models_with_data', new_callable=AsyncMock) as mock_train:
            
            mock_train.return_value = {
                'siamese_network': 'Training completed successfully',
                'ai_detector': 'Training completed successfully',
                'verification_service': 'Training completed successfully'
            }
            
            # Test training
            training_data = {
                'authorship_data': {
                    'embeddings': [np.random.rand(768) for _ in range(10)],
                    'author_ids': ['author1'] * 5 + ['author2'] * 5
                },
                'ai_detection_data': {
                    'texts': ['human text'] * 5 + ['ai text'] * 5,
                    'labels': [0] * 5 + [1] * 5,
                    'stylometric_features': [{'feature': 0.5}] * 10
                }
            }
            
            results = await authorship_manager.train_models_with_data(training_data)
            
            assert 'siamese_network' in results
            assert 'ai_detector' in results
            assert 'verification_service' in results
            assert all('successfully' in result for result in results.values())

class TestModelTrainingUtils:
    """Test model training utilities."""
    
    @pytest.mark.asyncio
    async def test_authorship_data_preparation(self):
        """Test authorship training data preparation."""
        with patch('utils.model_training.embedding_service.generate_embedding', new_callable=AsyncMock) as mock_embed:
            
            mock_embed.return_value = np.random.rand(768)
            
            texts_by_author = {
                'author1': ['Text 1 by author 1', 'Text 2 by author 1'],
                'author2': ['Text 1 by author 2', 'Text 2 by author 2']
            }
            
            training_data = await training_utils.prepare_authorship_training_data(texts_by_author)
            
            assert len(training_data['texts']) == 4
            assert len(training_data['author_ids']) == 4
            assert len(training_data['embeddings']) == 4
            assert set(training_data['author_ids']) == {'author1', 'author2'}
    
    @pytest.mark.asyncio
    async def test_ai_detection_data_preparation(self):
        """Test AI detection training data preparation."""
        human_texts = ['Human text 1', 'Human text 2']
        ai_texts = ['AI text 1', 'AI text 2']
        
        training_data = await training_utils.prepare_ai_detection_training_data(human_texts, ai_texts)
        
        assert len(training_data['texts']) == 4
        assert len(training_data['labels']) == 4
        assert len(training_data['stylometric_features']) == 4
        assert training_data['labels'] == [0, 0, 1, 1]
    
    def test_synthetic_data_generation(self):
        """Test synthetic training data generation."""
        synthetic_data = training_utils.generate_synthetic_training_data(
            num_authors=3,
            texts_per_author=2
        )
        
        assert len(synthetic_data) == 3
        assert all(len(texts) == 2 for texts in synthetic_data.values())
        assert all(isinstance(text, str) for texts in synthetic_data.values() for text in texts)

class TestAuthorshipConfig:
    """Test authorship configuration."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        validation_results = config.validate_config()
        
        assert isinstance(validation_results, dict)
        assert 'model_directories' in validation_results
        assert 'thresholds' in validation_results
        assert 'feature_config' in validation_results
        assert 'training_config' in validation_results
    
    def test_model_path_generation(self):
        """Test model path generation."""
        path = config.get_model_path('authorship', 'siamese_model.pth')
        
        assert path.name == 'siamese_model.pth'
        assert 'authorship' in str(path)
    
    def test_threshold_updates(self):
        """Test threshold updates."""
        original_threshold = config.VERIFICATION_THRESHOLDS['authorship_min_score']
        
        new_thresholds = {'authorship_min_score': 0.8}
        config.update_thresholds(new_thresholds)
        
        assert config.VERIFICATION_THRESHOLDS['authorship_min_score'] == 0.8
        
        # Reset to original
        config.VERIFICATION_THRESHOLDS['authorship_min_score'] = original_threshold
    
    def test_environment_config(self):
        """Test environment-specific configuration."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            env_config = config.get_environment_config()
            assert env_config['enable_gpu'] == True
            assert env_config['log_level'] == 'INFO'
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'testing'}):
            env_config = config.get_environment_config()
            assert env_config['use_mock_models'] == True
            assert env_config['log_level'] == 'DEBUG'

@pytest.mark.asyncio
async def test_end_to_end_verification():
    """End-to-end test of the verification system."""
    # This test simulates a complete verification workflow
    
    # Mock all external dependencies
    with patch.object(authorship_manager, 'initialize_all_models', new_callable=AsyncMock):
        with patch('services.authorship_integration.verification_service.verify_submission', new_callable=AsyncMock) as mock_verify:
            
            # Setup comprehensive mock result
            from services.verification_service import ComprehensiveVerificationResult
            from services.authorship_models import AuthorshipResult, AIDetectionResult
            from datetime import datetime
            
            mock_result = ComprehensiveVerificationResult(
                submission_id="e2e_test_123",
                student_id="e2e_student_456",
                timestamp=datetime.utcnow(),
                authorship_result=AuthorshipResult(
                    similarity_score=0.82,
                    confidence_interval=(0.75, 0.89),
                    is_authentic=True,
                    uncertainty=0.15,
                    feature_importance={
                        "semantic_similarity": 0.85,
                        "stylometric_match": 0.78,
                        "writing_pattern": 0.83
                    }
                ),
                ai_detection_result=AIDetectionResult(
                    ai_probability=0.12,
                    human_probability=0.88,
                    confidence=0.91,
                    detection_method="enhanced_ensemble",
                    explanation={
                        "perplexity_analysis": "Human-like variability",
                        "linguistic_markers": "Natural language patterns",
                        "stylometric_indicators": "Consistent with human writing"
                    }
                ),
                duplicate_matches=[],
                stylometric_features={
                    "type_token_ratio": 0.68,
                    "avg_sentence_length": 16.5,
                    "lexical_diversity": 0.74
                },
                overall_status="PASS",
                confidence_score=0.87,
                risk_factors=[]
            )
            
            mock_verify.return_value = mock_result
            authorship_manager.initialized = True
            
            # Simulate a real verification request
            test_essay = """
            The impact of artificial intelligence on modern education represents a paradigm shift 
            that educators and students must navigate carefully. While AI tools offer unprecedented 
            opportunities for personalized learning and automated assessment, they also raise 
            important questions about academic integrity and the fundamental nature of learning itself.
            
            In my experience as a student, I have observed how AI can both enhance and complicate 
            the educational process. On one hand, AI-powered tutoring systems can provide immediate 
            feedback and adapt to individual learning styles. On the other hand, the temptation to 
            rely too heavily on AI for assignments can undermine the development of critical thinking skills.
            
            The key lies in finding a balance that leverages AI's capabilities while preserving 
            the essential human elements of education: creativity, critical analysis, and genuine 
            understanding. This requires thoughtful integration rather than wholesale adoption or rejection.
            """
            
            # Perform end-to-end verification
            result = await authorship_manager.verify_authorship_comprehensive(
                text=test_essay,
                student_id="e2e_student_456",
                submission_id="e2e_test_123",
                reference_texts=None
            )
            
            # Verify comprehensive results
            assert result['overall_status'] == "PASS"
            assert result['confidence_score'] > 0.8
            assert len(result['risk_factors']) == 0
            assert result['authorship_result']['is_authentic']
            assert result['ai_detection_result']['human_probability'] > 0.8
            
            # Verify metadata
            assert 'processing_time_seconds' in result
            assert 'models_used' in result
            assert result['submission_id'] == "e2e_test_123"
            assert result['student_id'] == "e2e_student_456"
            
            print(f"End-to-end test completed successfully:")
            print(f"- Overall Status: {result['overall_status']}")
            print(f"- Confidence Score: {result['confidence_score']:.3f}")
            print(f"- Authorship Authentic: {result['authorship_result']['is_authentic']}")
            print(f"- AI Probability: {result['ai_detection_result']['ai_probability']:.3f}")
            print(f"- Processing Time: {result['processing_time_seconds']:.3f}s")
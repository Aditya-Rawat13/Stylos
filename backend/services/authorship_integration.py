"""
Integration service for authorship verification models with the existing system.
Provides high-level interfaces and model management.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from datetime import datetime
import json
from pathlib import Path

from .authorship_models import (
    authorship_service, 
    ai_detection_service,
    AuthorshipResult,
    AIDetectionResult
)
from .ai_detection_enhanced import enhanced_ai_detector
from .verification_service import verification_service
from .embedding_service import embedding_service
from .stylometric_analyzer import StylometricAnalyzer

logger = logging.getLogger(__name__)

class AuthorshipModelManager:
    """
    Manager for all authorship verification models and services.
    Provides unified interface for model operations.
    """
    
    def __init__(self):
        """Initialize the authorship model manager."""
        self.initialized = False
        self.model_status = {
            'siamese_network': False,
            'ai_detector': False,
            'enhanced_ai_detector': False,
            'embedding_service': False,
            'verification_service': False
        }
        
        # Performance metrics
        self.metrics = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'average_processing_time': 0.0,
            'model_accuracy': {}
        }
    
    async def initialize_all_models(self):
        """Initialize all authorship verification models."""
        try:
            logger.info("Initializing all authorship verification models...")
            
            # Initialize embedding service
            await embedding_service.initialize_models()
            self.model_status['embedding_service'] = True
            
            # Initialize authorship service
            await authorship_service.initialize()
            self.model_status['siamese_network'] = True
            
            # Initialize AI detection services
            await ai_detection_service.initialize_models()
            self.model_status['ai_detector'] = True
            
            await enhanced_ai_detector.initialize_models()
            self.model_status['enhanced_ai_detector'] = True
            
            # Initialize comprehensive verification service
            await verification_service.initialize()
            self.model_status['verification_service'] = True
            
            self.initialized = True
            logger.info("All authorship verification models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing authorship models: {e}")
            raise
    
    async def verify_authorship_comprehensive(self, text: str, student_id: str, 
                                           submission_id: str, 
                                           reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive authorship verification using all available models.
        
        Args:
            text: Text to verify
            student_id: Student identifier
            submission_id: Submission identifier
            reference_texts: Optional reference texts from the same author
            
        Returns:
            Comprehensive verification result
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.initialized:
                await self.initialize_all_models()
            
            # Create verification request
            from .verification_service import VerificationRequest
            request = VerificationRequest(
                text=text,
                student_id=student_id,
                submission_id=submission_id,
                reference_texts=reference_texts
            )
            
            # Perform comprehensive verification
            result = await verification_service.verify_submission(request)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(True, processing_time)
            
            # Convert to dictionary and add metadata
            result_dict = result.to_dict()
            result_dict['processing_time_seconds'] = processing_time
            result_dict['models_used'] = list(self.model_status.keys())
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in comprehensive authorship verification: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(False, processing_time)
            
            return {
                'submission_id': submission_id,
                'student_id': student_id,
                'error': str(e),
                'overall_status': 'ERROR',
                'processing_time_seconds': processing_time
            }
    
    async def quick_authorship_check(self, text: str, reference_embeddings: List[np.ndarray]) -> AuthorshipResult:
        """
        Perform quick authorship verification using only the Siamese network.
        
        Args:
            text: Text to verify
            reference_embeddings: Reference embeddings for comparison
            
        Returns:
            Authorship verification result
        """
        try:
            if not self.model_status['siamese_network']:
                await authorship_service.initialize()
            
            # Generate embedding for the text
            text_embedding = await embedding_service.generate_embedding(text)
            
            # Perform authorship verification
            result = await authorship_service.verify_authorship(text_embedding, reference_embeddings)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in quick authorship check: {e}")
            return AuthorshipResult(
                similarity_score=0.0,
                confidence_interval=(0.0, 0.0),
                is_authentic=False,
                uncertainty=1.0,
                feature_importance={"error": str(e)}
            )
    
    async def detect_ai_content_advanced(self, text: str) -> Dict[str, Any]:
        """
        Perform advanced AI content detection using enhanced models.
        
        Args:
            text: Text to analyze
            
        Returns:
            Advanced AI detection result
        """
        try:
            if not self.model_status['enhanced_ai_detector']:
                await enhanced_ai_detector.initialize_models()
            
            # Perform enhanced AI detection
            ai_prob, confidence, explanation = await enhanced_ai_detector.detect_ai_content_enhanced(text)
            
            return {
                'ai_probability': ai_prob,
                'human_probability': 1.0 - ai_prob,
                'confidence': confidence,
                'explanation': {
                    'primary_indicators': explanation.primary_indicators,
                    'confidence_factors': explanation.confidence_factors,
                    'risk_assessment': explanation.risk_assessment,
                    'summary': explanation.human_readable_summary
                },
                'detection_method': 'enhanced_ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error in advanced AI detection: {e}")
            return {
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'confidence': 0.0,
                'explanation': {'error': str(e)},
                'detection_method': 'error'
            }
    
    async def train_models_with_data(self, training_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Train all models with provided training data.
        
        Args:
            training_data: Training data for different models
            
        Returns:
            Training results for each model
        """
        results = {}
        
        try:
            # Train Siamese network
            if 'authorship_data' in training_data:
                authorship_data = training_data['authorship_data']
                embeddings = authorship_data.get('embeddings', [])
                author_ids = authorship_data.get('author_ids', [])
                
                if embeddings and author_ids:
                    await authorship_service.train_siamese_model(embeddings, author_ids)
                    results['siamese_network'] = 'Training completed successfully'
                else:
                    results['siamese_network'] = 'No training data provided'
            
            # Train AI detection models
            if 'ai_detection_data' in training_data:
                ai_data = training_data['ai_detection_data']
                texts = ai_data.get('texts', [])
                labels = ai_data.get('labels', [])
                stylometric_features = ai_data.get('stylometric_features', [])
                
                if texts and labels and stylometric_features:
                    await ai_detection_service.train(texts, labels, stylometric_features)
                    results['ai_detector'] = 'Training completed successfully'
                else:
                    results['ai_detector'] = 'No training data provided'
            
            # Train comprehensive verification service
            await verification_service.train_models(training_data)
            results['verification_service'] = 'Training completed successfully'
            
            logger.info("Model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            results['error'] = str(e)
            return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models and services."""
        return {
            'initialized': self.initialized,
            'model_status': self.model_status,
            'metrics': self.metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update performance metrics."""
        self.metrics['total_verifications'] += 1
        
        if success:
            self.metrics['successful_verifications'] += 1
        else:
            self.metrics['failed_verifications'] += 1
        
        # Update average processing time
        total_verifications = self.metrics['total_verifications']
        current_avg = self.metrics['average_processing_time']
        self.metrics['average_processing_time'] = (
            (current_avg * (total_verifications - 1) + processing_time) / total_verifications
        )
    
    async def benchmark_models(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmark model performance on test data.
        
        Args:
            test_data: Test data with ground truth labels
            
        Returns:
            Benchmark results for each model
        """
        try:
            benchmark_results = {}
            
            # Benchmark authorship verification
            if 'authorship_test' in test_data:
                auth_results = await self._benchmark_authorship(test_data['authorship_test'])
                benchmark_results['authorship_verification'] = auth_results
            
            # Benchmark AI detection
            if 'ai_detection_test' in test_data:
                ai_results = await self._benchmark_ai_detection(test_data['ai_detection_test'])
                benchmark_results['ai_detection'] = ai_results
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Error benchmarking models: {e}")
            return {'error': str(e)}
    
    async def _benchmark_authorship(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark authorship verification models."""
        try:
            texts = test_data.get('texts', [])
            reference_embeddings_list = test_data.get('reference_embeddings', [])
            ground_truth = test_data.get('ground_truth', [])
            
            if not all([texts, reference_embeddings_list, ground_truth]):
                return {'error': 'Incomplete test data'}
            
            correct_predictions = 0
            total_predictions = len(texts)
            
            for i, (text, ref_embeddings, true_label) in enumerate(zip(texts, reference_embeddings_list, ground_truth)):
                result = await self.quick_authorship_check(text, ref_embeddings)
                predicted_label = 1 if result.is_authentic else 0
                
                if predicted_label == true_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'total_samples': total_predictions,
                'correct_predictions': correct_predictions
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking authorship verification: {e}")
            return {'error': str(e)}
    
    async def _benchmark_ai_detection(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Benchmark AI detection models."""
        try:
            texts = test_data.get('texts', [])
            ground_truth = test_data.get('ground_truth', [])  # 0 for human, 1 for AI
            
            if not all([texts, ground_truth]):
                return {'error': 'Incomplete test data'}
            
            correct_predictions = 0
            total_predictions = len(texts)
            
            for text, true_label in zip(texts, ground_truth):
                result = await self.detect_ai_content_advanced(text)
                predicted_label = 1 if result['ai_probability'] > 0.5 else 0
                
                if predicted_label == true_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'total_samples': total_predictions,
                'correct_predictions': correct_predictions
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking AI detection: {e}")
            return {'error': str(e)}

# Global instance
authorship_manager = AuthorshipModelManager()
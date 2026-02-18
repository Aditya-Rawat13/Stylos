"""
Tests for enhanced AI detection system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.ai_detection_enhanced import (
    EnhancedAIDetectionClassifier,
    PerplexityAnalyzer,
    BurstinessAnalyzer,
    RepetitionPatternAnalyzer,
    LinguisticMarkerAnalyzer,
    AIDetectionFeatures,
    DetectionExplanation
)

class TestPerplexityAnalyzer:
    """Test cases for perplexity analyzer."""
    
    @pytest.fixture
    def perplexity_analyzer(self):
        """Create perplexity analyzer instance."""
        return PerplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_perplexity_analyzer_initialization(self, perplexity_analyzer):
        """Test perplexity analyzer initialization."""
        await perplexity_analyzer.initialize()
        
        # Should initialize without errors
        assert perplexity_analyzer.tokenizer is not None
        assert perplexity_analyzer.model is not None
    
    def test_perplexity_calculation(self, perplexity_analyzer):
        """Test perplexity calculation."""
        text = "This is a sample text for testing perplexity calculation."
        
        perplexity = perplexity_analyzer.calculate_perplexity(text)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert perplexity < 1000  # Reasonable upper bound

class TestBurstinessAnalyzer:
    """Test cases for burstiness analyzer."""
    
    @pytest.fixture
    def burstiness_analyzer(self):
        """Create burstiness analyzer instance."""
        return BurstinessAnalyzer()
    
    def test_burstiness_calculation_normal_text(self, burstiness_analyzer):
        """Test burstiness calculation with normal text."""
        text = "The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully."
        
        burstiness = burstiness_analyzer.calculate_burstiness(text)
        
        assert isinstance(burstiness, float)
        assert 0.0 <= burstiness <= 1.0
    
    def test_burstiness_calculation_repetitive_text(self, burstiness_analyzer):
        """Test burstiness calculation with repetitive text."""
        text = "The the the the the the the the the the."
        
        burstiness = burstiness_analyzer.calculate_burstiness(text)
        
        assert isinstance(burstiness, float)
        assert 0.0 <= burstiness <= 1.0
        # Repetitive text should have lower burstiness
        assert burstiness < 0.8
    
    def test_burstiness_calculation_short_text(self, burstiness_analyzer):
        """Test burstiness calculation with short text."""
        text = "Short text."
        
        burstiness = burstiness_analyzer.calculate_burstiness(text)
        
        assert burstiness == 0.5  # Should return neutral score for short texts

class TestRepetitionPatternAnalyzer:
    """Test cases for repetition pattern analyzer."""
    
    @pytest.fixture
    def repetition_analyzer(self):
        """Create repetition pattern analyzer instance."""
        return RepetitionPatternAnalyzer()
    
    def test_repetition_pattern_analysis(self, repetition_analyzer):
        """Test repetition pattern analysis."""
        text = "Furthermore, it is important to note that this is a test. Moreover, this demonstrates repetition."
        
        patterns = repetition_analyzer.analyze_repetition_patterns(text)
        
        assert isinstance(patterns, dict)
        assert 'sentence_structure_repetition' in patterns
        assert 'bigram_repetition' in patterns
        assert 'trigram_repetition' in patterns
        assert 'phrase_repetition' in patterns
        assert 'transition_overuse' in patterns
        
        # All scores should be between 0 and 1
        for score in patterns.values():
            assert 0.0 <= score <= 1.0
    
    def test_sentence_structure_repetition(self, repetition_analyzer):
        """Test sentence structure repetition detection."""
        # Sentences with similar lengths
        sentences = [
            "This is a test sentence",
            "Here is another test sentence", 
            "This is yet another sentence"
        ]
        
        repetition = repetition_analyzer._calculate_sentence_structure_repetition(sentences)
        
        assert isinstance(repetition, float)
        assert 0.0 <= repetition <= 1.0
    
    def test_ngram_repetition(self, repetition_analyzer):
        """Test n-gram repetition calculation."""
        words = ["the", "quick", "brown", "fox", "the", "quick", "brown", "dog"]
        
        bigram_repetition = repetition_analyzer._calculate_ngram_repetition(words, 2)
        trigram_repetition = repetition_analyzer._calculate_ngram_repetition(words, 3)
        
        assert isinstance(bigram_repetition, float)
        assert isinstance(trigram_repetition, float)
        assert 0.0 <= bigram_repetition <= 1.0
        assert 0.0 <= trigram_repetition <= 1.0
        
        # Should detect repetition in this example
        assert bigram_repetition > 0
    
    def test_phrase_repetition(self, repetition_analyzer):
        """Test phrase repetition detection."""
        text = "It is important to note that this is a test. Furthermore, it is important to note the results."
        
        repetition = repetition_analyzer._calculate_phrase_repetition(text)
        
        assert isinstance(repetition, float)
        assert 0.0 <= repetition <= 1.0
        # Should detect the repeated phrase
        assert repetition > 0
    
    def test_transition_overuse(self, repetition_analyzer):
        """Test transition word overuse detection."""
        words = ["however", "the", "test", "furthermore", "shows", "moreover", "that", "results"]
        
        overuse = repetition_analyzer._calculate_transition_overuse(words)
        
        assert isinstance(overuse, float)
        assert 0.0 <= overuse <= 1.0
        # Should detect overuse of transition words
        assert overuse > 0

class TestLinguisticMarkerAnalyzer:
    """Test cases for linguistic marker analyzer."""
    
    @pytest.fixture
    def linguistic_analyzer(self):
        """Create linguistic marker analyzer instance."""
        return LinguisticMarkerAnalyzer()
    
    def test_linguistic_marker_analysis(self, linguistic_analyzer):
        """Test linguistic marker analysis."""
        text = "It is important to utilize these methods to facilitate the implementation of the system."
        
        markers = linguistic_analyzer.analyze_linguistic_markers(text)
        
        assert isinstance(markers, dict)
        assert 'excessive_formality' in markers
        assert 'hedging_language' in markers
        assert 'generic_language' in markers
        assert 'lack_personal_experience' in markers
        assert 'overly_structured' in markers
        
        # All scores should be between 0 and 1
        for score in markers.values():
            assert 0.0 <= score <= 1.0
    
    def test_formality_score(self, linguistic_analyzer):
        """Test formality score calculation."""
        formal_text = "We must utilize these methods to facilitate the implementation and demonstrate the results."
        informal_text = "We should use these ways to help set up and show the results."
        
        formal_score = linguistic_analyzer._calculate_formality_score(formal_text)
        informal_score = linguistic_analyzer._calculate_formality_score(informal_text)
        
        assert isinstance(formal_score, float)
        assert isinstance(informal_score, float)
        assert formal_score > informal_score  # Formal text should have higher score
    
    def test_hedging_score(self, linguistic_analyzer):
        """Test hedging language score calculation."""
        hedging_text = "This might possibly be the case, and it could potentially work."
        direct_text = "This is the case, and it will work."
        
        hedging_score = linguistic_analyzer._calculate_hedging_score(hedging_text)
        direct_score = linguistic_analyzer._calculate_hedging_score(direct_text)
        
        assert isinstance(hedging_score, float)
        assert isinstance(direct_score, float)
        assert hedging_score > direct_score  # Hedging text should have higher score
    
    def test_generic_language_score(self, linguistic_analyzer):
        """Test generic language score calculation."""
        generic_text = "There are various aspects and different ways to consider the important factors."
        specific_text = "The algorithm processes data using neural networks and gradient descent."
        
        generic_score = linguistic_analyzer._calculate_generic_language_score(generic_text)
        specific_score = linguistic_analyzer._calculate_generic_language_score(specific_text)
        
        assert isinstance(generic_score, float)
        assert isinstance(specific_score, float)
        assert generic_score > specific_score  # Generic text should have higher score
    
    def test_personal_experience_score(self, linguistic_analyzer):
        """Test personal experience score calculation."""
        impersonal_text = "The research shows that the method is effective."
        personal_text = "I think the method works well based on my experience."
        
        impersonal_score = linguistic_analyzer._calculate_personal_experience_score(impersonal_text)
        personal_score = linguistic_analyzer._calculate_personal_experience_score(personal_text)
        
        assert isinstance(impersonal_score, float)
        assert isinstance(personal_score, float)
        assert impersonal_score > personal_score  # Impersonal text should have higher score
    
    def test_structure_score(self, linguistic_analyzer):
        """Test structure score calculation."""
        structured_text = "Firstly, we examine the data. Secondly, we analyze the results. Finally, we draw conclusions."
        unstructured_text = "We look at the data and see what happens. The results are interesting."
        
        structured_score = linguistic_analyzer._calculate_structure_score(structured_text)
        unstructured_score = linguistic_analyzer._calculate_structure_score(unstructured_text)
        
        assert isinstance(structured_score, float)
        assert isinstance(unstructured_score, float)
        assert structured_score > unstructured_score  # Structured text should have higher score

class TestEnhancedAIDetectionClassifier:
    """Test cases for enhanced AI detection classifier."""
    
    @pytest.fixture
    def enhanced_detector(self):
        """Create enhanced AI detection classifier instance."""
        return EnhancedAIDetectionClassifier()
    
    @pytest.mark.asyncio
    async def test_enhanced_detector_initialization(self, enhanced_detector):
        """Test enhanced detector initialization."""
        await enhanced_detector.initialize_models()
        
        # Should initialize without errors
        assert enhanced_detector.perplexity_analyzer is not None
        assert enhanced_detector.burstiness_analyzer is not None
        assert enhanced_detector.repetition_analyzer is not None
        assert enhanced_detector.linguistic_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_feature_extraction(self, enhanced_detector):
        """Test comprehensive feature extraction."""
        await enhanced_detector.initialize_models()
        
        text = "This is a comprehensive test text for feature extraction. It contains multiple sentences with various patterns."
        
        features = await enhanced_detector.extract_comprehensive_features(text)
        
        assert isinstance(features, AIDetectionFeatures)
        assert isinstance(features.perplexity_score, float)
        assert isinstance(features.burstiness_score, float)
        assert isinstance(features.repetition_patterns, dict)
        assert isinstance(features.linguistic_markers, dict)
        assert isinstance(features.stylometric_anomalies, dict)
        assert isinstance(features.transformer_embeddings, np.ndarray)
        
        # Validate score ranges
        assert features.perplexity_score > 0
        assert 0.0 <= features.burstiness_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_enhanced_ai_detection(self, enhanced_detector):
        """Test enhanced AI detection process."""
        await enhanced_detector.initialize_models()
        
        # Test with potentially AI-generated text (formal, structured)
        ai_like_text = """
        Furthermore, it is important to note that artificial intelligence represents a significant advancement 
        in computational technology. Moreover, the implementation of machine learning algorithms facilitates 
        the development of sophisticated systems. Additionally, these systems demonstrate remarkable capabilities 
        in various applications.
        """
        
        ai_prob, confidence, explanation = await enhanced_detector.detect_ai_content_enhanced(ai_like_text)
        
        assert isinstance(ai_prob, float)
        assert isinstance(confidence, float)
        assert isinstance(explanation, DetectionExplanation)
        
        assert 0.0 <= ai_prob <= 1.0
        assert 0.0 <= confidence <= 1.0
        
        # Validate explanation structure
        assert isinstance(explanation.primary_indicators, list)
        assert isinstance(explanation.confidence_factors, dict)
        assert isinstance(explanation.linguistic_evidence, dict)
        assert isinstance(explanation.model_contributions, dict)
        assert isinstance(explanation.risk_assessment, str)
        assert isinstance(explanation.human_readable_summary, str)
    
    @pytest.mark.asyncio
    async def test_human_vs_ai_text_detection(self, enhanced_detector):
        """Test detection difference between human-like and AI-like text."""
        await enhanced_detector.initialize_models()
        
        # Human-like text (personal, varied, less formal)
        human_text = """
        I've been thinking about this problem for weeks now. Yesterday, I had a breakthrough while 
        walking my dog. The solution isn't what I expected - it's actually much simpler. 
        My colleague Sarah disagrees, but I think she's missing the point.
        """
        
        # AI-like text (formal, structured, generic)
        ai_text = """
        It is important to consider the various aspects of this problem. Furthermore, the implementation 
        of appropriate solutions requires careful analysis. Moreover, the systematic approach facilitates 
        optimal results. In conclusion, these methods demonstrate significant effectiveness.
        """
        
        human_prob, human_conf, human_exp = await enhanced_detector.detect_ai_content_enhanced(human_text)
        ai_prob, ai_conf, ai_exp = await enhanced_detector.detect_ai_content_enhanced(ai_text)
        
        # AI-like text should have higher AI probability
        assert ai_prob > human_prob
        
        # Both should have reasonable confidence
        assert human_conf > 0.1
        assert ai_conf > 0.1
    
    def test_stylometric_anomaly_calculation(self, enhanced_detector):
        """Test stylometric anomaly calculation."""
        # Text with low vocabulary diversity
        repetitive_text = "The cat sat on the mat. The cat sat on the hat. The cat sat on the bat."
        
        anomalies = enhanced_detector._calculate_stylometric_anomalies(repetitive_text)
        
        assert isinstance(anomalies, dict)
        assert 'low_vocab_diversity' in anomalies
        assert 'sentence_uniformity' in anomalies
        assert 'punctuation_anomaly' in anomalies
        
        # Should detect low vocabulary diversity
        assert anomalies['low_vocab_diversity'] > 0
        
        # Should detect sentence uniformity
        assert anomalies['sentence_uniformity'] > 0
    
    def test_perplexity_scoring(self, enhanced_detector):
        """Test perplexity to AI probability scoring."""
        # Low perplexity should indicate higher AI probability
        low_perplexity_score = enhanced_detector._score_perplexity(15.0)
        high_perplexity_score = enhanced_detector._score_perplexity(80.0)
        
        assert isinstance(low_perplexity_score, float)
        assert isinstance(high_perplexity_score, float)
        assert 0.0 <= low_perplexity_score <= 1.0
        assert 0.0 <= high_perplexity_score <= 1.0
        
        # Lower perplexity should result in higher AI probability
        assert low_perplexity_score > high_perplexity_score

@pytest.mark.asyncio
async def test_integration_enhanced_ai_detection():
    """Integration test for enhanced AI detection system."""
    detector = EnhancedAIDetectionClassifier()
    
    # Mock initialization to avoid loading actual models
    with patch.object(detector.perplexity_analyzer, 'initialize', new_callable=AsyncMock):
        with patch.object(detector, '_initialize_transformer_models', new_callable=AsyncMock):
            await detector.initialize_models()
    
    # Test with sample texts
    test_cases = [
        {
            "text": "I personally believe this approach works well based on my experience.",
            "expected_ai_prob": "low",  # Human-like
            "description": "Personal, experiential text"
        },
        {
            "text": "Furthermore, it is essential to implement systematic methodologies to facilitate optimal outcomes.",
            "expected_ai_prob": "high",  # AI-like
            "description": "Formal, generic text"
        },
        {
            "text": "The quick brown fox jumps over the lazy dog repeatedly and consistently.",
            "expected_ai_prob": "medium",  # Neutral
            "description": "Simple, clear text"
        }
    ]
    
    for case in test_cases:
        ai_prob, confidence, explanation = await detector.detect_ai_content_enhanced(case["text"])
        
        # Validate basic properties
        assert 0.0 <= ai_prob <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert len(explanation.human_readable_summary) > 0
        
        # Log results for manual verification
        print(f"\nTest case: {case['description']}")
        print(f"Text: {case['text'][:50]}...")
        print(f"AI Probability: {ai_prob:.3f}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Summary: {explanation.human_readable_summary}")
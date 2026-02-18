"""
Unit tests for stylometric analyzer service.
Tests lexical richness, sentence analysis, and feature extraction.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.stylometric_analyzer import StylometricAnalyzer


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return StylometricAnalyzer()


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a test essay. It contains multiple sentences with varying lengths.
    The purpose is to test stylometric analysis. We need diverse content here.
    Some sentences are short. Others are significantly longer and more complex.
    Punctuation matters! Does it work? Yes, it does.
    """


def test_lexical_diversity_calculation(analyzer, sample_text):
    """Test lexical diversity (TTR) calculation."""
    features = analyzer.extract_features(sample_text)
    
    assert 'lexical_diversity' in features
    assert 0 <= features['lexical_diversity'] <= 1
    assert features['lexical_diversity'] > 0


def test_sentence_length_analysis(analyzer, sample_text):
    """Test average sentence length calculation."""
    features = analyzer.extract_features(sample_text)
    
    assert 'avg_sentence_length' in features
    assert features['avg_sentence_length'] > 0
    assert isinstance(features['avg_sentence_length'], (int, float))


def test_punctuation_frequency(analyzer, sample_text):
    """Test punctuation pattern analysis."""
    features = analyzer.extract_features(sample_text)
    
    assert 'punctuation_frequency' in features
    assert isinstance(features['punctuation_frequency'], dict)
    assert '.' in features['punctuation_frequency']
    assert '!' in features['punctuation_frequency']
    assert '?' in features['punctuation_frequency']


def test_empty_text_handling(analyzer):
    """Test handling of empty text."""
    features = analyzer.extract_features("")
    
    assert features is not None
    assert features['lexical_diversity'] == 0
    assert features['avg_sentence_length'] == 0


def test_single_sentence_text(analyzer):
    """Test analysis of single sentence."""
    text = "This is a single sentence."
    features = analyzer.extract_features(text)
    
    assert features['avg_sentence_length'] > 0
    assert features['lexical_diversity'] > 0


def test_feature_consistency(analyzer, sample_text):
    """Test that same text produces consistent features."""
    features1 = analyzer.extract_features(sample_text)
    features2 = analyzer.extract_features(sample_text)
    
    assert features1['lexical_diversity'] == features2['lexical_diversity']
    assert features1['avg_sentence_length'] == features2['avg_sentence_length']


def test_pos_tag_distribution(analyzer, sample_text):
    """Test POS tag distribution analysis."""
    features = analyzer.extract_features(sample_text)
    
    if 'pos_tag_distribution' in features:
        assert isinstance(features['pos_tag_distribution'], dict)
        assert len(features['pos_tag_distribution']) > 0


def test_function_word_frequency(analyzer, sample_text):
    """Test function word frequency analysis."""
    features = analyzer.extract_features(sample_text)
    
    if 'function_word_frequency' in features:
        assert isinstance(features['function_word_frequency'], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

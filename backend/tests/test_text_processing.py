"""
Unit tests for text processing utilities.
Tests text extraction, cleaning, and preprocessing.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.text_processor import TextProcessor


@pytest.fixture
def processor():
    """Create text processor instance."""
    return TextProcessor()


def test_text_cleaning(processor):
    """Test text cleaning and normalization."""
    dirty_text = "  This   has   extra    spaces.  \n\n\n Multiple newlines.  "
    cleaned = processor.clean_text(dirty_text)
    
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.strip() == cleaned


def test_tokenization(processor):
    """Test text tokenization."""
    text = "This is a test sentence. Another sentence here."
    tokens = processor.tokenize(text)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "test" in tokens or "This" in tokens


def test_sentence_segmentation(processor):
    """Test sentence segmentation."""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = processor.segment_sentences(text)
    
    assert isinstance(sentences, list)
    assert len(sentences) == 3


def test_minimum_length_validation(processor):
    """Test minimum text length validation."""
    short_text = "Too short."
    long_text = "This is a sufficiently long text that should pass validation checks. " * 10
    
    assert not processor.validate_length(short_text, min_words=50)
    assert processor.validate_length(long_text, min_words=50)


def test_special_character_removal(processor):
    """Test removal of special characters."""
    text_with_special = "Hello @#$% World! Test 123."
    cleaned = processor.remove_special_chars(text_with_special)
    
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "$" not in cleaned


def test_text_quality_validation(processor):
    """Test overall text quality validation."""
    good_text = "This is a well-formed essay with proper sentences. " * 20
    bad_text = "asdfjkl qwerty zxcvbn"
    
    assert processor.validate_quality(good_text)
    assert not processor.validate_quality(bad_text)


def test_empty_text_handling(processor):
    """Test handling of empty or None text."""
    assert processor.clean_text("") == ""
    assert processor.tokenize("") == []
    assert processor.segment_sentences("") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

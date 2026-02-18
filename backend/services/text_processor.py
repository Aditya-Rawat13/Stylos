"""
Text processing service for cleaning and preparing text for analysis.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextProcessor:
    """Service for text preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize text processor."""
        pass
    
    async def process_text(self, text: str) -> str:
        """
        Process and clean text for analysis.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and processed text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def extract_text_from_file(self, content: bytes, content_type: str) -> str:
        """
        Extract text from file content based on content type.
        
        Args:
            content: File content as bytes
            content_type: MIME type of the file
            
        Returns:
            Extracted text
        """
        if content_type == 'text/plain':
            return content.decode('utf-8', errors='ignore')
        elif content_type == 'application/pdf':
            from services.file_processor import file_processor
            return file_processor.extract_text_from_pdf(content)
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            from services.file_processor import file_processor
            return file_processor.extract_text_from_docx(content)
        else:
            # Try to decode as text
            try:
                return content.decode('utf-8', errors='ignore')
            except:
                return ""
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def get_text_stats(self, text: str) -> dict:
        """Get basic statistics about the text."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }


# Global instance
text_processor = TextProcessor()

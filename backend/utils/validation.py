"""
Data validation and sanitization utilities.
"""
import re
import html
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status


class DataValidator:
    """Data validation and sanitization utilities."""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input by removing harmful content."""
        if not isinstance(text, str):
            return ""
        
        # HTML escape
        text = html.escape(text)
        
        # Remove potential script tags and other dangerous content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_text_length(text: str, min_length: int = 0, max_length: int = 10000) -> bool:
        """Validate text length."""
        return min_length <= len(text) <= max_length
    
    @staticmethod
    def validate_word_count(text: str, min_words: int = 0, max_words: int = 10000) -> bool:
        """Validate word count in text."""
        words = text.split()
        return min_words <= len(words) <= max_words
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\.\.', '', filename)  # Remove parent directory references
        filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def validate_institution_id(institution_id: str) -> bool:
        """Validate institution ID format."""
        # Allow alphanumeric, hyphens, and underscores
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, institution_id)) and len(institution_id) <= 100
    
    @staticmethod
    def validate_student_id(student_id: str) -> bool:
        """Validate student ID format."""
        # Allow alphanumeric, hyphens, and underscores
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, student_id)) and len(student_id) <= 100
    
    @staticmethod
    def validate_essay_content(content: str) -> Dict[str, Any]:
        """Validate essay content and return analysis."""
        if not content or not isinstance(content, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Essay content is required"
            )
        
        # Basic length validation
        if len(content) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Essay content must be at least 100 characters long"
            )
        
        if len(content) > 50000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Essay content must not exceed 50,000 characters"
            )
        
        # Word count validation
        words = content.split()
        if len(words) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Essay must contain at least 50 words"
            )
        
        if len(words) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Essay must not exceed 10,000 words"
            )
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'lorem ipsum',  # Placeholder text
            r'test test test',  # Repeated test text
            r'(.)\1{20,}',  # Repeated characters
            r'(\w+\s+)\1{10,}',  # Repeated words
        ]
        
        warnings = []
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append(f"Suspicious pattern detected: {pattern}")
        
        # Language detection (basic)
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        english_word_count = sum(1 for word in words if word.lower() in english_words)
        english_ratio = english_word_count / len(words) if words else 0
        
        if english_ratio < 0.05:
            warnings.append("Content may not be in English")
        
        return {
            'character_count': len(content),
            'word_count': len(words),
            'english_ratio': english_ratio,
            'warnings': warnings,
            'is_valid': len(warnings) == 0
        }
    
    @staticmethod
    def sanitize_json_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON data recursively."""
        if isinstance(data, dict):
            return {key: DataValidator.sanitize_json_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [DataValidator.sanitize_json_data(item) for item in data]
        elif isinstance(data, str):
            return DataValidator.sanitize_text(data)
        else:
            return data
    
    @staticmethod
    def validate_profile_data(profile_data: Dict[str, Any]) -> List[str]:
        """Validate writing profile data and return any issues."""
        issues = []
        
        # Check required fields
        required_fields = ['lexical_features', 'syntactic_features', 'semantic_features']
        for field in required_fields:
            if field not in profile_data or not profile_data[field]:
                issues.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if 'total_words' in profile_data:
            total_words = profile_data['total_words']
            if not isinstance(total_words, int) or total_words < 0:
                issues.append("Total words must be a non-negative integer")
        
        if 'avg_confidence_score' in profile_data:
            confidence = profile_data['avg_confidence_score']
            if confidence is not None and (not isinstance(confidence, int) or not 0 <= confidence <= 100):
                issues.append("Confidence score must be between 0 and 100")
        
        return issues


# Global validator instance
validator = DataValidator()
"""
User and writing profile schemas.
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

from models.user import UserRole


class UserProfileUpdate(BaseModel):
    """User profile update schema."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    institution_id: Optional[str] = Field(None, max_length=100)
    student_id: Optional[str] = Field(None, max_length=100)


class WritingProfileResponse(BaseModel):
    """Writing profile response schema."""
    id: int
    studentId: int
    stylometricFeatures: Dict[str, Any]
    semanticEmbedding: List[float] = []
    confidenceScore: float
    sampleCount: int
    lastUpdated: str
    createdAt: str
    statistics: Dict[str, Any]
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, obj):
        """Create response from ORM object with frontend-compatible structure."""
        # Transform backend structure to frontend structure
        lexical = obj.lexical_features or {}
        syntactic = obj.syntactic_features or {}
        semantic = obj.semantic_features or {}
        
        # Build stylometric features in the format frontend expects
        stylometric_features = {
            'lexicalRichness': {
                'ttr': lexical.get('type_token_ratio', 0.5),
                'mtld': lexical.get('vocabulary_richness', 50.0),
                'vocdD': lexical.get('unique_words', 100.0)
            },
            'syntacticComplexity': {
                'avgSentenceLength': syntactic.get('avg_sentence_length', 20.0),
                'avgClauseLength': syntactic.get('avg_sentence_length', 20.0) * 0.6,  # Estimate
                'subordinationRatio': 0.3  # Default value
            },
            'punctuationPatterns': {
                'commaFrequency': syntactic.get('punctuation_frequency', {}).get('comma', 0.02),
                'semicolonFrequency': syntactic.get('punctuation_frequency', {}).get('semicolon', 0.001),
                'exclamationFrequency': syntactic.get('punctuation_frequency', {}).get('exclamation', 0.001),
                'questionFrequency': syntactic.get('punctuation_frequency', {}).get('question', 0.005)
            },
            'wordFrequencies': {
                'functionWords': semantic.get('function_word_frequency', {}),
                'contentWords': {}
            },
            'posTagDistribution': {
                'NOUN': 0.25,
                'VERB': 0.20,
                'ADJ': 0.15,
                'ADV': 0.10,
                'PRON': 0.08,
                'DET': 0.08,
                'ADP': 0.08,
                'CONJ': 0.06
            }
        }
        
        data = {
            'id': str(obj.id),
            'studentId': str(obj.user_id),
            'stylometricFeatures': stylometric_features,
            'semanticEmbedding': [],
            'confidenceScore': obj.avg_confidence_score / 100.0 if obj.avg_confidence_score else 0.0,
            'sampleCount': obj.total_submissions,
            'lastUpdated': obj.last_updated.isoformat() if obj.last_updated else obj.created_at.isoformat(),
            'createdAt': obj.created_at.isoformat(),
            'statistics': {
                'totalSubmissions': obj.total_submissions,
                'averageLength': obj.total_words // max(obj.total_submissions, 1),
                'topicDistribution': {},
                'timePatterns': {
                    'preferredWritingHours': [],
                    'averageWritingTime': 0
                },
                'improvementMetrics': {
                    'authorshipConsistency': obj.avg_confidence_score / 100.0 if obj.avg_confidence_score else 0.0,
                    'styleEvolution': [],
                    'qualityTrend': []
                }
            }
        }
        return cls(**data)


class WritingProfileStats(BaseModel):
    """Writing profile statistics schema."""
    lexical_diversity: Optional[float]
    avg_sentence_length: Optional[float]
    vocabulary_richness: Optional[float]
    punctuation_frequency: Optional[Dict[str, float]]
    pos_distribution: Optional[Dict[str, float]]
    function_word_frequency: Optional[Dict[str, float]]


class SampleEssayUpload(BaseModel):
    """Sample essay upload for profile initialization."""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=100, max_length=50000)
    
    @validator('content')
    def validate_content_length(cls, v):
        word_count = len(v.split())
        if word_count < 50:
            raise ValueError('Essay must contain at least 50 words')
        if word_count > 10000:
            raise ValueError('Essay must not exceed 10,000 words')
        return v


class WritingProfileInitRequest(BaseModel):
    """Writing profile initialization request."""
    sample_essays: List[SampleEssayUpload] = Field(..., min_items=1, max_items=5)
    
    @validator('sample_essays')
    def validate_sample_count(cls, v):
        if len(v) < 1:
            raise ValueError('At least 1 sample essay is required')
        if len(v) > 5:
            raise ValueError('Maximum 5 sample essays allowed')
        return v


class UserProfileResponse(BaseModel):
    """Complete user profile response."""
    id: int
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    institution_id: Optional[str]
    student_id: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]
    writing_profile: Optional[WritingProfileResponse]
    
    class Config:
        from_attributes = True


class WritingProfileUpdateRequest(BaseModel):
    """Writing profile update request."""
    additional_essays: List[SampleEssayUpload] = Field(..., min_items=1, max_items=3)


class ProfileAnalysisResponse(BaseModel):
    """Profile analysis response."""
    profile_strength: float  # 0.0 to 1.0
    recommendations: List[str]
    missing_features: List[str]
    confidence_level: str  # "low", "medium", "high"
    sample_count: int
    total_words: int
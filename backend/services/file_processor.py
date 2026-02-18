"""
File processing service for extracting text from uploaded files.
"""
import os
import hashlib
import uuid
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException
import aiofiles
from docx import Document
import PyPDF2
import io


class FileProcessor:
    """Service for processing uploaded files and extracting text content."""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR = "uploads"
    
    def __init__(self):
        # Create upload directory if it doesn't exist
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
    
    def validate_file(self, filename: str, content_size: int) -> None:
        """Validate uploaded file."""
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        if content_size > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {self.MAX_FILE_SIZE // (1024*1024)}MB"
            )
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from DOCX: {str(e)}"
            )
    
    def extract_text_from_txt(self, content: bytes) -> str:
        """Extract text from TXT file."""
        try:
            return content.decode('utf-8').strip()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                return content.decode('latin-1').strip()
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read text file: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from TXT: {str(e)}"
            )
    
    def extract_text(self, content: bytes, filename: str) -> str:
        """Extract text based on file extension."""
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(content)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(content)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(content)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}"
            )
    
    def validate_text_content(self, text: str) -> None:
        """Validate extracted text content."""
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text content found in the uploaded file"
            )
        
        word_count = len(text.split())
        if word_count < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Text too short. Minimum 50 words required, found {word_count} words"
            )
        
        if word_count > 10000:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long. Maximum 10,000 words allowed, found {word_count} words"
            )
    
    def generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content for duplicate detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


# Global instance
file_processor = FileProcessor()


async def extract_text_from_file(content: bytes, filename: str) -> str:
    """
    Extract text from uploaded file content.
    
    Args:
        content: File content as bytes
        filename: Original filename
        
    Returns:
        Extracted text content
    """
    file_processor.validate_file(filename, len(content))
    text = file_processor.extract_text(content, filename)
    file_processor.validate_text_content(text)
    return text

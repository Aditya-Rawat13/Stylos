"""
Encryption utilities for data protection and privacy.
Implements AES-256 encryption for sensitive data storage.
"""
import os
import base64
import hashlib
import secrets
from typing import Optional, Tuple, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for encrypting and decrypting sensitive data."""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryption service with master key."""
        self.master_key = master_key or os.getenv("ENCRYPTION_MASTER_KEY")
        if not self.master_key:
            raise ValueError("Encryption master key is required")
        
        # Derive encryption key from master key
        self._encryption_key = self._derive_key(self.master_key.encode())
    
    def _derive_key(self, password: bytes, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = b"stylos_salt_2024"  # Static salt for consistent key derivation
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password)
    
    def encrypt_text(self, plaintext: str) -> str:
        """Encrypt text using AES-256-GCM."""
        try:
            # Generate random IV
            iv = os.urandom(12)  # 96-bit IV for GCM
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self._encryption_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()
            
            # Combine IV + auth_tag + ciphertext and encode as base64
            encrypted_data = iv + encryptor.tag + ciphertext
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text using AES-256-GCM."""
        try:
            # Decode from base64
            encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
            
            # Extract components
            iv = encrypted_data[:12]  # First 12 bytes
            auth_tag = encrypted_data[12:28]  # Next 16 bytes
            ciphertext = encrypted_data[28:]  # Remaining bytes
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self._encryption_key),
                modes.GCM(iv, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_file_content(self, file_content: bytes) -> bytes:
        """Encrypt file content using AES-256-GCM."""
        try:
            # Generate random IV
            iv = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self._encryption_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(file_content) + encryptor.finalize()
            
            # Combine IV + auth_tag + ciphertext
            return iv + encryptor.tag + ciphertext
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise
    
    def decrypt_file_content(self, encrypted_content: bytes) -> bytes:
        """Decrypt file content using AES-256-GCM."""
        try:
            # Extract components
            iv = encrypted_content[:12]
            auth_tag = encrypted_content[12:28]
            ciphertext = encrypted_content[28:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self._encryption_key),
                modes.GCM(iv, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            return decryptor.update(ciphertext) + decryptor.finalize()
            
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash sensitive data with salt for secure storage."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use SHA-256 with salt
        hash_obj = hashlib.sha256()
        hash_obj.update(salt.encode('utf-8'))
        hash_obj.update(data.encode('utf-8'))
        
        return hash_obj.hexdigest(), salt
    
    def verify_hashed_data(self, data: str, hashed_data: str, salt: str) -> bool:
        """Verify data against its hash."""
        computed_hash, _ = self.hash_sensitive_data(data, salt)
        return secrets.compare_digest(computed_hash, hashed_data)


class DataAnonymizer:
    """Service for anonymizing sensitive data for analytics."""
    
    @staticmethod
    def anonymize_email(email: str) -> str:
        """Anonymize email address for analytics."""
        if '@' not in email:
            return "anonymous@example.com"
        
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            anonymized_local = "**"
        else:
            anonymized_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{anonymized_local}@{domain}"
    
    @staticmethod
    def anonymize_ip_address(ip_address: str) -> str:
        """Anonymize IP address by masking last octet."""
        if not ip_address:
            return "0.0.0.0"
        
        # Handle IPv4
        if '.' in ip_address and ':' not in ip_address:
            parts = ip_address.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
        
        # Handle IPv6 (simplified)
        if ':' in ip_address:
            parts = ip_address.split(':')
            if len(parts) >= 4:
                return ':'.join(parts[:4]) + '::0'
        
        return "0.0.0.0"
    
    @staticmethod
    def anonymize_user_agent(user_agent: str) -> str:
        """Anonymize user agent by keeping only browser family."""
        if not user_agent:
            return "Unknown"
        
        # Extract browser family (simplified)
        user_agent_lower = user_agent.lower()
        if 'chrome' in user_agent_lower:
            return "Chrome"
        elif 'firefox' in user_agent_lower:
            return "Firefox"
        elif 'safari' in user_agent_lower:
            return "Safari"
        elif 'edge' in user_agent_lower:
            return "Edge"
        else:
            return "Other"
    
    @staticmethod
    def generate_pseudonym(user_id: int, salt: str = "stylos_pseudonym") -> str:
        """Generate consistent pseudonym for user."""
        hash_obj = hashlib.sha256()
        hash_obj.update(salt.encode('utf-8'))
        hash_obj.update(str(user_id).encode('utf-8'))
        
        # Generate readable pseudonym
        hash_hex = hash_obj.hexdigest()
        return f"User_{hash_hex[:8]}"


# Global encryption service instance
encryption_service = None

def get_encryption_service() -> EncryptionService:
    """Get global encryption service instance."""
    global encryption_service
    if encryption_service is None:
        encryption_service = EncryptionService()
    return encryption_service
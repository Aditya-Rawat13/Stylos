"""
IPFS service for decentralized storage of essays and metadata.
"""
import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import aiohttp
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


class IPFSEncryption:
    """Handles encryption/decryption for sensitive content on IPFS."""
    
    def __init__(self, password: str):
        """Initialize encryption with password-derived key."""
        self.password = password.encode()
        self._fernet = None
    
    def _get_fernet(self) -> Fernet:
        """Get or create Fernet cipher instance."""
        if self._fernet is None:
            salt = b'stylos_ipfs_salt'  # In production, use random salt per file
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password))
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt string data."""
        fernet = self._get_fernet()
        return fernet.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data back to string."""
        fernet = self._get_fernet()
        return fernet.decrypt(encrypted_data).decode()


class IPFSClient:
    """IPFS client for content storage and retrieval."""
    
    def __init__(self):
        self.api_url = settings.IPFS_API_URL
        self.gateway_url = settings.IPFS_GATEWAY_URL
        self.api_key = settings.IPFS_API_KEY
        self.api_secret = settings.IPFS_API_SECRET
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Initialize encryption
        self.encryption = IPFSEncryption(settings.IPFS_ENCRYPTION_PASSWORD)
        
        # Pin tracking
        self.pinned_hashes: Dict[str, datetime] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get authenticated IPFS session."""
        auth = aiohttp.BasicAuth(self.api_key, self.api_secret) if self.api_key else None
        return aiohttp.ClientSession(
            timeout=self.timeout,
            auth=auth,
            headers={'User-Agent': 'Stylos-IPFS-Client/1.0'}
        )
    
    async def add_content(
        self, 
        content: str, 
        filename: str = None,
        encrypt: bool = True,
        pin: bool = True
    ) -> str:
        """
        Add content to IPFS.
        
        Args:
            content: Text content to store
            filename: Optional filename for the content
            encrypt: Whether to encrypt the content
            pin: Whether to pin the content
            
        Returns:
            IPFS hash of the stored content
        """
        try:
            # Prepare content
            if encrypt:
                content_bytes = self.encryption.encrypt(content)
                content_type = 'application/octet-stream'
            else:
                content_bytes = content.encode('utf-8')
                content_type = 'text/plain'
            
            # Create form data
            data = aiohttp.FormData()
            data.add_field(
                'file',
                content_bytes,
                filename=filename or 'content.txt',
                content_type=content_type
            )
            
            # Add to IPFS
            async with await self._get_session() as session:
                async with session.post(f"{self.api_url}/api/v0/add", data=data) as response:
                    if response.status != 200:
                        raise Exception(f"IPFS add failed: {response.status}")
                    
                    result = await response.json()
                    ipfs_hash = result['Hash']
                    
                    logger.info(f"Content added to IPFS: {ipfs_hash}")
                    
                    # Pin the content if requested
                    if pin:
                        await self.pin_content(ipfs_hash)
                    
                    return ipfs_hash
                    
        except Exception as e:
            logger.error(f"Failed to add content to IPFS: {e}")
            raise
    
    async def get_content(self, ipfs_hash: str, decrypt: bool = True) -> str:
        """
        Retrieve content from IPFS.
        
        Args:
            ipfs_hash: IPFS hash of the content
            decrypt: Whether to decrypt the content
            
        Returns:
            Retrieved content as string
        """
        try:
            async with await self._get_session() as session:
                url = f"{self.gateway_url}/{ipfs_hash}"
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"IPFS get failed: {response.status}")
                    
                    content_bytes = await response.read()
                    
                    if decrypt:
                        try:
                            content = self.encryption.decrypt(content_bytes)
                        except Exception:
                            # If decryption fails, assume it's plain text
                            content = content_bytes.decode('utf-8')
                    else:
                        content = content_bytes.decode('utf-8')
                    
                    logger.info(f"Content retrieved from IPFS: {ipfs_hash}")
                    return content
                    
        except Exception as e:
            logger.error(f"Failed to get content from IPFS: {e}")
            raise
    
    async def pin_content(self, ipfs_hash: str) -> bool:
        """
        Pin content to prevent garbage collection.
        
        Args:
            ipfs_hash: IPFS hash to pin
            
        Returns:
            True if pinning successful
        """
        try:
            async with await self._get_session() as session:
                url = f"{self.api_url}/api/v0/pin/add"
                params = {'arg': ipfs_hash}
                
                async with session.post(url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"IPFS pin failed: {response.status}")
                    
                    self.pinned_hashes[ipfs_hash] = datetime.utcnow()
                    logger.info(f"Content pinned: {ipfs_hash}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to pin content: {e}")
            return False
    
    async def unpin_content(self, ipfs_hash: str) -> bool:
        """
        Unpin content to allow garbage collection.
        
        Args:
            ipfs_hash: IPFS hash to unpin
            
        Returns:
            True if unpinning successful
        """
        try:
            async with await self._get_session() as session:
                url = f"{self.api_url}/api/v0/pin/rm"
                params = {'arg': ipfs_hash}
                
                async with session.post(url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"IPFS unpin failed: {response.status}")
                    
                    self.pinned_hashes.pop(ipfs_hash, None)
                    logger.info(f"Content unpinned: {ipfs_hash}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to unpin content: {e}")
            return False
    
    async def get_pinned_content(self) -> List[Dict[str, Any]]:
        """
        Get list of pinned content.
        
        Returns:
            List of pinned content information
        """
        try:
            async with await self._get_session() as session:
                url = f"{self.api_url}/api/v0/pin/ls"
                
                async with session.post(url) as response:
                    if response.status != 200:
                        raise Exception(f"IPFS pin list failed: {response.status}")
                    
                    result = await response.json()
                    pins = []
                    
                    for hash_info in result.get('Keys', {}):
                        pins.append({
                            'hash': hash_info,
                            'type': result['Keys'][hash_info]['Type'],
                            'pinned_at': self.pinned_hashes.get(hash_info)
                        })
                    
                    return pins
                    
        except Exception as e:
            logger.error(f"Failed to get pinned content: {e}")
            return []


class IPFSService:
    """High-level IPFS service for essay and metadata storage."""
    
    def __init__(self):
        self.client = IPFSClient()
        self.redundancy_nodes = settings.IPFS_REDUNDANCY_NODES or []
    
    async def store_essay(
        self, 
        essay_content: str, 
        metadata: Dict[str, Any],
        submission_id: str
    ) -> Tuple[str, str]:
        """
        Store essay content and metadata on IPFS.
        
        Args:
            essay_content: Full essay text
            metadata: Essay metadata (verification results, etc.)
            submission_id: Unique submission identifier
            
        Returns:
            Tuple of (content_hash, metadata_hash)
        """
        try:
            # Store essay content (encrypted)
            content_filename = f"essay_{submission_id}.txt"
            content_hash = await self.client.add_content(
                essay_content,
                filename=content_filename,
                encrypt=True,
                pin=True
            )
            
            # Prepare metadata with content reference
            full_metadata = {
                'submission_id': submission_id,
                'content_hash': content_hash,
                'stored_at': datetime.utcnow().isoformat(),
                'content_size': len(essay_content),
                'encrypted': True,
                **metadata
            }
            
            # Store metadata (not encrypted for transparency)
            metadata_filename = f"metadata_{submission_id}.json"
            metadata_json = json.dumps(full_metadata, indent=2)
            metadata_hash = await self.client.add_content(
                metadata_json,
                filename=metadata_filename,
                encrypt=False,
                pin=True
            )
            
            # Store on redundancy nodes if configured
            await self._replicate_to_redundancy_nodes(content_hash, metadata_hash)
            
            logger.info(f"Essay stored on IPFS - Content: {content_hash}, Metadata: {metadata_hash}")
            
            return content_hash, metadata_hash
            
        except Exception as e:
            logger.error(f"Failed to store essay on IPFS: {e}")
            raise
    
    async def retrieve_essay(self, content_hash: str) -> str:
        """
        Retrieve essay content from IPFS.
        
        Args:
            content_hash: IPFS hash of the essay content
            
        Returns:
            Essay content as string
        """
        try:
            content = await self.client.get_content(content_hash, decrypt=True)
            logger.info(f"Essay retrieved from IPFS: {content_hash}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve essay from IPFS: {e}")
            # Try redundancy nodes
            for node_url in self.redundancy_nodes:
                try:
                    # Implement redundancy retrieval logic here
                    pass
                except Exception:
                    continue
            raise
    
    async def retrieve_metadata(self, metadata_hash: str) -> Dict[str, Any]:
        """
        Retrieve metadata from IPFS.
        
        Args:
            metadata_hash: IPFS hash of the metadata
            
        Returns:
            Metadata dictionary
        """
        try:
            metadata_json = await self.client.get_content(metadata_hash, decrypt=False)
            metadata = json.loads(metadata_json)
            logger.info(f"Metadata retrieved from IPFS: {metadata_hash}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve metadata from IPFS: {e}")
            raise
    
    async def update_essay_metadata(
        self, 
        original_metadata_hash: str,
        updates: Dict[str, Any]
    ) -> str:
        """
        Update essay metadata by creating new version.
        
        Args:
            original_metadata_hash: Hash of original metadata
            updates: Updates to apply to metadata
            
        Returns:
            New metadata hash
        """
        try:
            # Retrieve original metadata
            original_metadata = await self.retrieve_metadata(original_metadata_hash)
            
            # Apply updates
            updated_metadata = {
                **original_metadata,
                **updates,
                'updated_at': datetime.utcnow().isoformat(),
                'previous_version': original_metadata_hash
            }
            
            # Store updated metadata
            submission_id = original_metadata.get('submission_id', 'unknown')
            metadata_filename = f"metadata_{submission_id}_updated.json"
            metadata_json = json.dumps(updated_metadata, indent=2)
            
            new_metadata_hash = await self.client.add_content(
                metadata_json,
                filename=metadata_filename,
                encrypt=False,
                pin=True
            )
            
            logger.info(f"Metadata updated on IPFS: {new_metadata_hash}")
            return new_metadata_hash
            
        except Exception as e:
            logger.error(f"Failed to update metadata on IPFS: {e}")
            raise
    
    async def _replicate_to_redundancy_nodes(self, *hashes: str):
        """Replicate content to redundancy nodes."""
        if not self.redundancy_nodes:
            return
        
        for node_url in self.redundancy_nodes:
            try:
                # Implement replication logic for each redundancy node
                # This would involve pinning content on multiple IPFS nodes
                pass
            except Exception as e:
                logger.warning(f"Failed to replicate to node {node_url}: {e}")
    
    async def cleanup_old_content(self, days_old: int = 30):
        """
        Clean up old unpinned content.
        
        Args:
            days_old: Number of days after which to consider content old
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Get pinned content
            pinned_content = await self.client.get_pinned_content()
            
            for pin_info in pinned_content:
                pinned_at = pin_info.get('pinned_at')
                if pinned_at and pinned_at < cutoff_date:
                    # Check if this is still referenced in database
                    # If not, unpin it
                    # This would require database integration
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old content: {e}")
    
    async def check_connection(self) -> bool:
        """
        Check IPFS node connectivity.
        
        Returns:
            True if IPFS node is accessible
        """
        try:
            async with await self.client._get_session() as session:
                url = f"{self.client.api_url}/api/v0/version"
                async with session.post(url) as response:
                    if response.status == 200:
                        version_info = await response.json()
                        logger.info(f"IPFS node connected - Version: {version_info.get('Version', 'unknown')}")
                        return True
                    else:
                        logger.warning(f"IPFS connection failed with status: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"IPFS connection check failed: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get IPFS storage statistics.
        
        Returns:
            Storage statistics dictionary
        """
        try:
            pinned_content = await self.client.get_pinned_content()
            
            stats = {
                'total_pinned_items': len(pinned_content),
                'pinned_content': pinned_content,
                'redundancy_nodes': len(self.redundancy_nodes),
                'encryption_enabled': True,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}


# Global IPFS service instance
ipfs_service = IPFSService()
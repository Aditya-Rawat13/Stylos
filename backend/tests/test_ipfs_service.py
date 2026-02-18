"""
Tests for IPFS service functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

from services.ipfs_service import IPFSService, IPFSClient, IPFSEncryption


class TestIPFSEncryption:
    """Test IPFS encryption functionality."""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test encryption and decryption roundtrip."""
        encryption = IPFSEncryption("test-password")
        original_text = "This is a test essay content for encryption."
        
        # Encrypt
        encrypted_data = encryption.encrypt(original_text)
        assert isinstance(encrypted_data, bytes)
        assert encrypted_data != original_text.encode()
        
        # Decrypt
        decrypted_text = encryption.decrypt(encrypted_data)
        assert decrypted_text == original_text
    
    def test_different_passwords_produce_different_results(self):
        """Test that different passwords produce different encrypted results."""
        text = "Test content"
        
        encryption1 = IPFSEncryption("password1")
        encryption2 = IPFSEncryption("password2")
        
        encrypted1 = encryption1.encrypt(text)
        encrypted2 = encryption2.encrypt(text)
        
        assert encrypted1 != encrypted2
    
    def test_wrong_password_fails_decryption(self):
        """Test that wrong password fails decryption."""
        text = "Test content"
        
        encryption1 = IPFSEncryption("correct-password")
        encryption2 = IPFSEncryption("wrong-password")
        
        encrypted = encryption1.encrypt(text)
        
        with pytest.raises(Exception):
            encryption2.decrypt(encrypted)


class TestIPFSClient:
    """Test IPFS client functionality."""
    
    @pytest.fixture
    def ipfs_client(self):
        """Create IPFS client for testing."""
        with patch('services.ipfs_service.settings') as mock_settings:
            mock_settings.IPFS_API_URL = "http://localhost:5001"
            mock_settings.IPFS_GATEWAY_URL = "http://localhost:8080/ipfs"
            mock_settings.IPFS_API_KEY = "test-key"
            mock_settings.IPFS_API_SECRET = "test-secret"
            mock_settings.IPFS_ENCRYPTION_PASSWORD = "test-password"
            
            return IPFSClient()
    
    @pytest.mark.asyncio
    async def test_add_content_success(self, ipfs_client):
        """Test successful content addition to IPFS."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"Hash": "QmTestHash123"})
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(ipfs_client, 'pin_content', return_value=True):
                result = await ipfs_client.add_content("test content", "test.txt")
                
                assert result == "QmTestHash123"
                mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_content_failure(self, ipfs_client):
        """Test content addition failure."""
        mock_response = Mock()
        mock_response.status = 500
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(Exception, match="IPFS add failed"):
                await ipfs_client.add_content("test content")
    
    @pytest.mark.asyncio
    async def test_get_content_success(self, ipfs_client):
        """Test successful content retrieval from IPFS."""
        test_content = "This is test content"
        encrypted_content = ipfs_client.encryption.encrypt(test_content)
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=encrypted_content)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await ipfs_client.get_content("QmTestHash123")
            
            assert result == test_content
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_content_plain_text(self, ipfs_client):
        """Test retrieval of plain text content."""
        test_content = "Plain text content"
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=test_content.encode())
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await ipfs_client.get_content("QmTestHash123", decrypt=False)
            
            assert result == test_content
    
    @pytest.mark.asyncio
    async def test_pin_content_success(self, ipfs_client):
        """Test successful content pinning."""
        mock_response = Mock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await ipfs_client.pin_content("QmTestHash123")
            
            assert result is True
            assert "QmTestHash123" in ipfs_client.pinned_hashes
    
    @pytest.mark.asyncio
    async def test_unpin_content_success(self, ipfs_client):
        """Test successful content unpinning."""
        # First add to pinned hashes
        ipfs_client.pinned_hashes["QmTestHash123"] = datetime.utcnow()
        
        mock_response = Mock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await ipfs_client.unpin_content("QmTestHash123")
            
            assert result is True
            assert "QmTestHash123" not in ipfs_client.pinned_hashes


class TestIPFSService:
    """Test high-level IPFS service functionality."""
    
    @pytest.fixture
    def ipfs_service(self):
        """Create IPFS service for testing."""
        with patch('services.ipfs_service.settings') as mock_settings:
            mock_settings.IPFS_REDUNDANCY_NODES = []
            return IPFSService()
    
    @pytest.mark.asyncio
    async def test_store_essay_success(self, ipfs_service):
        """Test successful essay storage."""
        essay_content = "This is a test essay about machine learning."
        metadata = {
            "authorship_score": 85,
            "ai_probability": 15,
            "verification_status": "PASS"
        }
        submission_id = "12345"
        
        # Mock the client methods
        with patch.object(ipfs_service.client, 'add_content') as mock_add:
            mock_add.side_effect = ["QmContentHash123", "QmMetadataHash456"]
            
            content_hash, metadata_hash = await ipfs_service.store_essay(
                essay_content, metadata, submission_id
            )
            
            assert content_hash == "QmContentHash123"
            assert metadata_hash == "QmMetadataHash456"
            assert mock_add.call_count == 2
            
            # Check that content was encrypted and metadata was not
            calls = mock_add.call_args_list
            assert calls[0][1]['encrypt'] is True  # Content encrypted
            assert calls[1][1]['encrypt'] is False  # Metadata not encrypted
    
    @pytest.mark.asyncio
    async def test_retrieve_essay_success(self, ipfs_service):
        """Test successful essay retrieval."""
        expected_content = "Retrieved essay content"
        
        with patch.object(ipfs_service.client, 'get_content') as mock_get:
            mock_get.return_value = expected_content
            
            result = await ipfs_service.retrieve_essay("QmContentHash123")
            
            assert result == expected_content
            mock_get.assert_called_once_with("QmContentHash123", decrypt=True)
    
    @pytest.mark.asyncio
    async def test_retrieve_metadata_success(self, ipfs_service):
        """Test successful metadata retrieval."""
        metadata = {
            "submission_id": "12345",
            "authorship_score": 85,
            "stored_at": "2023-01-01T00:00:00"
        }
        metadata_json = json.dumps(metadata)
        
        with patch.object(ipfs_service.client, 'get_content') as mock_get:
            mock_get.return_value = metadata_json
            
            result = await ipfs_service.retrieve_metadata("QmMetadataHash456")
            
            assert result == metadata
            mock_get.assert_called_once_with("QmMetadataHash456", decrypt=False)
    
    @pytest.mark.asyncio
    async def test_update_essay_metadata(self, ipfs_service):
        """Test metadata update functionality."""
        original_metadata = {
            "submission_id": "12345",
            "authorship_score": 85,
            "stored_at": "2023-01-01T00:00:00"
        }
        
        updates = {
            "authorship_score": 90,
            "verification_status": "UPDATED"
        }
        
        with patch.object(ipfs_service, 'retrieve_metadata') as mock_retrieve:
            mock_retrieve.return_value = original_metadata
            
            with patch.object(ipfs_service.client, 'add_content') as mock_add:
                mock_add.return_value = "QmNewMetadataHash789"
                
                result = await ipfs_service.update_essay_metadata(
                    "QmOriginalHash", updates
                )
                
                assert result == "QmNewMetadataHash789"
                
                # Verify the updated metadata includes both original and new data
                call_args = mock_add.call_args[0]
                updated_metadata_json = call_args[0]
                updated_metadata = json.loads(updated_metadata_json)
                
                assert updated_metadata["authorship_score"] == 90
                assert updated_metadata["verification_status"] == "UPDATED"
                assert updated_metadata["submission_id"] == "12345"
                assert "updated_at" in updated_metadata
                assert updated_metadata["previous_version"] == "QmOriginalHash"
    
    @pytest.mark.asyncio
    async def test_get_storage_stats(self, ipfs_service):
        """Test storage statistics retrieval."""
        mock_pinned_content = [
            {
                "hash": "QmHash1",
                "type": "recursive",
                "pinned_at": datetime.utcnow()
            },
            {
                "hash": "QmHash2",
                "type": "direct",
                "pinned_at": datetime.utcnow()
            }
        ]
        
        with patch.object(ipfs_service.client, 'get_pinned_content') as mock_get_pinned:
            mock_get_pinned.return_value = mock_pinned_content
            
            stats = await ipfs_service.get_storage_stats()
            
            assert stats["total_pinned_items"] == 2
            assert stats["pinned_content"] == mock_pinned_content
            assert stats["redundancy_nodes"] == 0
            assert stats["encryption_enabled"] is True
            assert "last_updated" in stats
    
    @pytest.mark.asyncio
    async def test_store_essay_with_redundancy(self, ipfs_service):
        """Test essay storage with redundancy nodes."""
        # Configure redundancy nodes
        ipfs_service.redundancy_nodes = ["http://node1:5001", "http://node2:5001"]
        
        essay_content = "Test essay for redundancy"
        metadata = {"test": "metadata"}
        submission_id = "test-123"
        
        with patch.object(ipfs_service.client, 'add_content') as mock_add:
            mock_add.side_effect = ["QmContent", "QmMetadata"]
            
            with patch.object(ipfs_service, '_replicate_to_redundancy_nodes') as mock_replicate:
                await ipfs_service.store_essay(essay_content, metadata, submission_id)
                
                # Verify replication was called
                mock_replicate.assert_called_once_with("QmContent", "QmMetadata")


@pytest.mark.asyncio
async def test_ipfs_service_error_handling():
    """Test error handling in IPFS service."""
    with patch('services.ipfs_service.settings') as mock_settings:
        mock_settings.IPFS_REDUNDANCY_NODES = []
        service = IPFSService()
        
        # Test error in store_essay
        with patch.object(service.client, 'add_content') as mock_add:
            mock_add.side_effect = Exception("IPFS connection failed")
            
            with pytest.raises(Exception, match="IPFS connection failed"):
                await service.store_essay("content", {}, "123")
        
        # Test error in retrieve_essay
        with patch.object(service.client, 'get_content') as mock_get:
            mock_get.side_effect = Exception("Content not found")
            
            with pytest.raises(Exception, match="Content not found"):
                await service.retrieve_essay("QmHash")


if __name__ == "__main__":
    pytest.main([__file__])
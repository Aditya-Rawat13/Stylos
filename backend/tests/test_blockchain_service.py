"""
Tests for blockchain service functionality.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from web3 import Web3

from services.blockchain_service import BlockchainService
from models.blockchain import BlockchainRecord, BlockchainStatus


class TestBlockchainService:
    """Test blockchain service functionality."""
    
    @pytest.fixture
    def blockchain_service(self):
        """Create blockchain service for testing."""
        with patch('services.blockchain_service.settings') as mock_settings:
            mock_settings.POLYGON_RPC_URL = "http://localhost:8545"
            mock_settings.PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS = "0x1234567890123456789012345678901234567890"
            mock_settings.BLOCKCHAIN_PRIVATE_KEY = "0x" + "1" * 64
            
            service = BlockchainService()
            
            # Mock Web3 and contract
            service.w3 = Mock()
            service.contract = Mock()
            service.account = Mock()
            service.account.address = "0xTestAddress"
            
            return service
    
    @pytest.mark.asyncio
    async def test_create_blockchain_attestation_success(self, blockchain_service):
        """Test successful blockchain attestation creation."""
        submission_id = 123
        essay_content = "This is a test essay for blockchain attestation."
        verification_results = {
            "authorship_score": 85,
            "ai_probability": 15,
            "overall_status": "PASS"
        }
        student_address = "0xStudentAddress"
        
        # Mock IPFS service
        with patch('services.blockchain_service.ipfs_service') as mock_ipfs:
            mock_ipfs.store_essay.return_value = ("QmContentHash", "QmMetadataHash")
            
            # Mock transaction submission
            with patch.object(blockchain_service, '_submit_mint_transaction') as mock_submit:
                mock_submit.return_value = "0xTransactionHash"
                
                result = await blockchain_service.create_blockchain_attestation(
                    submission_id,
                    essay_content,
                    verification_results,
                    student_address
                )
                
                assert isinstance(result, BlockchainRecord)
                assert result.submission_id == submission_id
                assert result.transaction_hash == "0xTransactionHash"
                assert result.status == BlockchainStatus.SUBMITTED
                assert result.ipfs_hash == "QmContentHash"
                assert result.ipfs_metadata_hash == "QmMetadataHash"
                assert result.authorship_score == 85
    
    @pytest.mark.asyncio
    async def test_create_blockchain_attestation_failure(self, blockchain_service):
        """Test blockchain attestation creation failure."""
        submission_id = 123
        essay_content = "Test essay"
        verification_results = {"authorship_score": 85}
        student_address = "0xStudentAddress"
        
        # Mock IPFS service failure
        with patch('services.blockchain_service.ipfs_service') as mock_ipfs:
            mock_ipfs.store_essay.side_effect = Exception("IPFS storage failed")
            
            result = await blockchain_service.create_blockchain_attestation(
                submission_id,
                essay_content,
                verification_results,
                student_address
            )
            
            assert result.status == BlockchainStatus.FAILED
            assert "IPFS storage failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_submit_mint_transaction(self, blockchain_service):
        """Test mint transaction submission."""
        record = BlockchainRecord(
            submission_id=123,
            content_hash="abcd1234",
            authorship_score=85,
            ipfs_hash="QmTestHash"
        )
        student_address = "0xStudentAddress"
        
        # Mock Web3 methods
        blockchain_service.w3.eth.get_transaction_count.return_value = 5
        blockchain_service.w3.to_wei.return_value = 30000000000
        blockchain_service.w3.eth.account.sign_transaction.return_value = Mock(rawTransaction=b"signed_tx")
        blockchain_service.w3.eth.send_raw_transaction.return_value = Mock(hex=lambda: "0xTxHash")
        
        # Mock contract function
        mock_function = Mock()
        mock_function.build_transaction.return_value = {
            "to": "0xContractAddress",
            "data": "0xdata"
        }
        blockchain_service.contract.functions.mintProofToken.return_value = mock_function
        
        result = await blockchain_service._submit_mint_transaction(
            record, student_address, "university", "course"
        )
        
        assert result == "0xTxHash"
    
    @pytest.mark.asyncio
    async def test_check_transaction_status_confirmed(self, blockchain_service):
        """Test checking confirmed transaction status."""
        tx_hash = "0xTransactionHash"
        
        # Mock transaction receipt
        mock_receipt = {
            "status": 1,
            "blockNumber": 12345,
            "blockHash": Mock(hex=lambda: "0xBlockHash"),
            "gasUsed": 150000,
            "logs": []
        }
        
        blockchain_service.w3.eth.get_transaction_receipt.return_value = mock_receipt
        blockchain_service.w3.eth.block_number = 12350
        
        result = await blockchain_service.check_transaction_status(tx_hash)
        
        assert result["status"] == "CONFIRMED"
        assert result["block_number"] == 12345
        assert result["gas_used"] == 150000
        assert result["confirmations"] == 5
    
    @pytest.mark.asyncio
    async def test_check_transaction_status_failed(self, blockchain_service):
        """Test checking failed transaction status."""
        tx_hash = "0xTransactionHash"
        
        # Mock failed transaction receipt
        mock_receipt = {
            "status": 0,
            "blockNumber": 12345,
            "blockHash": Mock(hex=lambda: "0xBlockHash"),
            "gasUsed": 21000,
            "logs": []
        }
        
        blockchain_service.w3.eth.get_transaction_receipt.return_value = mock_receipt
        blockchain_service.w3.eth.block_number = 12350
        
        result = await blockchain_service.check_transaction_status(tx_hash)
        
        assert result["status"] == "FAILED"
        assert result["block_number"] == 12345
    
    @pytest.mark.asyncio
    async def test_check_transaction_status_pending(self, blockchain_service):
        """Test checking pending transaction status."""
        tx_hash = "0xTransactionHash"
        
        # Mock no receipt (pending)
        blockchain_service.w3.eth.get_transaction_receipt.return_value = None
        
        result = await blockchain_service.check_transaction_status(tx_hash)
        
        assert result["status"] == "PENDING"
        assert result["confirmations"] == 0
    
    @pytest.mark.asyncio
    async def test_update_blockchain_record_status(self, blockchain_service):
        """Test updating blockchain record status."""
        record = BlockchainRecord(
            submission_id=123,
            transaction_hash="0xTxHash",
            status=BlockchainStatus.SUBMITTED
        )
        
        # Mock confirmed transaction
        with patch.object(blockchain_service, 'check_transaction_status') as mock_check:
            mock_check.return_value = {
                "status": "CONFIRMED",
                "block_number": 12345,
                "block_hash": "0xBlockHash",
                "gas_used": 150000,
                "token_id": "1"
            }
            
            updated_record = await blockchain_service.update_blockchain_record_status(record)
            
            assert updated_record.status == BlockchainStatus.CONFIRMED
            assert updated_record.block_number == 12345
            assert updated_record.token_id == "1"
            assert updated_record.confirmed_at is not None
    
    @pytest.mark.asyncio
    async def test_get_student_tokens(self, blockchain_service):
        """Test getting student tokens from contract."""
        student_address = "0xStudentAddress"
        
        # Mock contract calls
        blockchain_service.contract.functions.getStudentSubmissions.return_value.call.return_value = [1, 2, 3]
        
        mock_submission_data = [
            b"content_hash_1",  # contentHash
            b"stylometric_hash_1",  # stylometricHash
            "QmIpfsHash1",  # ipfsHash
            "0xStudentAddress",  # student
            1640995200,  # timestamp
            85,  # authorshipScore
            15,  # aiProbability
            True,  # verified
            "university-123",  # institutionId
            "cs-101"  # courseId
        ]
        
        blockchain_service.contract.functions.getSubmission.return_value.call.return_value = mock_submission_data
        
        tokens = await blockchain_service.get_student_tokens(student_address)
        
        assert len(tokens) == 3
        for token in tokens:
            assert token["authorship_score"] == 85
            assert token["ai_probability"] == 15
            assert token["verified"] is True
    
    @pytest.mark.asyncio
    async def test_get_network_stats(self, blockchain_service):
        """Test getting network statistics."""
        # Mock latest block
        mock_block = {
            "number": 12345,
            "timestamp": 1640995200
        }
        blockchain_service.w3.eth.get_block.return_value = mock_block
        blockchain_service.w3.eth.gas_price = 30000000000
        blockchain_service.w3.from_wei.return_value = 30.0
        
        stats = await blockchain_service.get_network_stats()
        
        assert stats["network"] == "polygon"
        assert stats["block_height"] == 12345
        assert stats["gas_price"] == 30.0
        assert stats["network_status"] == "HEALTHY"
    
    @pytest.mark.asyncio
    async def test_retry_failed_attestation(self, blockchain_service):
        """Test retrying failed blockchain attestation."""
        record = BlockchainRecord(
            submission_id=123,
            status=BlockchainStatus.FAILED,
            retry_count=1,
            max_retries=3
        )
        
        updated_record = await blockchain_service.retry_failed_attestation(record)
        
        assert updated_record.retry_count == 2
        assert updated_record.status == BlockchainStatus.PENDING
        assert updated_record.error_message is None
    
    @pytest.mark.asyncio
    async def test_retry_failed_attestation_max_retries(self, blockchain_service):
        """Test retry when max retries exceeded."""
        record = BlockchainRecord(
            submission_id=123,
            status=BlockchainStatus.FAILED,
            retry_count=3,
            max_retries=3
        )
        
        updated_record = await blockchain_service.retry_failed_attestation(record)
        
        # Should not retry when max retries exceeded
        assert updated_record.retry_count == 3
        assert updated_record.status == BlockchainStatus.FAILED
    
    def test_generate_content_hash(self, blockchain_service):
        """Test content hash generation."""
        content = "Test essay content"
        hash_result = blockchain_service._generate_content_hash(content)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex length
        
        # Same content should produce same hash
        hash_result2 = blockchain_service._generate_content_hash(content)
        assert hash_result == hash_result2
        
        # Different content should produce different hash
        different_hash = blockchain_service._generate_content_hash("Different content")
        assert hash_result != different_hash
    
    def test_generate_stylometric_hash(self, blockchain_service):
        """Test stylometric hash generation."""
        submission_id = 123
        hash_result = blockchain_service._generate_stylometric_hash(submission_id)
        
        assert isinstance(hash_result, bytes)
        assert len(hash_result) == 32  # SHA-256 byte length
        
        # Same ID should produce same hash
        hash_result2 = blockchain_service._generate_stylometric_hash(submission_id)
        assert hash_result == hash_result2
        
        # Different ID should produce different hash
        different_hash = blockchain_service._generate_stylometric_hash(456)
        assert hash_result != different_hash


@pytest.mark.asyncio
async def test_blockchain_service_initialization():
    """Test blockchain service initialization."""
    with patch('services.blockchain_service.settings') as mock_settings:
        mock_settings.POLYGON_RPC_URL = "http://localhost:8545"
        mock_settings.PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS = "0x1234"
        mock_settings.BLOCKCHAIN_PRIVATE_KEY = "0x" + "1" * 64
        
        with patch('services.blockchain_service.Web3') as mock_web3:
            service = BlockchainService()
            
            # Verify Web3 was initialized
            mock_web3.assert_called_once()
            
            # Verify middleware was added
            assert service.w3.middleware_onion.inject.called


@pytest.mark.asyncio
async def test_blockchain_service_no_private_key():
    """Test blockchain service initialization without private key."""
    with patch('services.blockchain_service.settings') as mock_settings:
        mock_settings.POLYGON_RPC_URL = "http://localhost:8545"
        mock_settings.PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS = "0x1234"
        mock_settings.BLOCKCHAIN_PRIVATE_KEY = None
        
        with patch('services.blockchain_service.Web3'):
            service = BlockchainService()
            
            assert service.account is None


if __name__ == "__main__":
    pytest.main([__file__])
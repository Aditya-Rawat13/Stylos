"""
Blockchain service for interacting with ProofOfAuthorship smart contract.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
from web3 import Web3, AsyncWeb3
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
from eth_account import Account
import aiohttp

from core.config import settings
from services.ipfs_service import ipfs_service
from models.blockchain import BlockchainRecord, BlockchainStatus

logger = logging.getLogger(__name__)


class BlockchainService:
    """Service for blockchain interactions and smart contract management."""
    
    def __init__(self):
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(settings.POLYGON_RPC_URL))
        
        # Add PoA middleware only for Polygon networks (not needed for Hardhat)
        network_name = getattr(settings, 'NETWORK_NAME', 'localhost')
        if network_name in ['polygon', 'mumbai']:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Load contract ABI and address
        self.contract_address = settings.PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS
        self.contract_abi = self._load_contract_abi()
        
        # Initialize contract instance
        if self.contract_address and self.contract_abi:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
        else:
            self.contract = None
            logger.warning("Contract not initialized - missing address or ABI")
        
        # Account for signing transactions
        if settings.BLOCKCHAIN_PRIVATE_KEY:
            self.account = Account.from_key(settings.BLOCKCHAIN_PRIVATE_KEY)
        else:
            self.account = None
            logger.warning("No private key configured for blockchain transactions")
    
    
    async def get_demo_portfolio_tokens(self, student_address: str) -> List[Dict[str, Any]]:
        """Get demo tokens for portfolio display when blockchain is offline."""
        try:
            from core.database import AsyncSessionLocal
            from models.user import User
            from models.submission import Submission
            from models.blockchain import BlockchainRecord
            from sqlalchemy import select
            
            async with AsyncSessionLocal() as db:
                # Find user by wallet address
                user_stmt = select(User).filter(User.wallet_address == student_address)
                user_result = await db.execute(user_stmt)
                user = user_result.scalar_one_or_none()
                
                if not user:
                    return []
                
                # Get user's blockchain records using raw SQL to avoid model issues
                from sqlalchemy import text
                stmt = text("""
                    SELECT br.*, s.title, s.content 
                    FROM blockchain_records br 
                    JOIN submissions s ON br.submission_id = s.id 
                    WHERE s.user_id = :user_id AND br.status = 'CONFIRMED'
                    ORDER BY br.created_at DESC
                    LIMIT 10
                """)
                result = await db.execute(stmt, {'user_id': user.id})
                records = result.fetchall()
                
                tokens = []
                for record in records:
                    token_info = {
                        'token_id': record.token_id or str(1000 + record.id),
                        'submission_id': str(record.submission_id),
                        'submission_title': record.title or f"Submission {record.submission_id}",
                        'minted_at': record.confirmed_at if record.confirmed_at else datetime.utcnow(),
                        'transaction_hash': record.transaction_hash,
                        'ipfs_metadata': {
                            'name': f"Proof of Authorship - {record.title or f'Submission {record.submission_id}'}",
                            'description': f"Academic authorship verification for submission {record.submission_id}",
                            'attributes': [
                                {'trait_type': 'Authorship Score', 'value': record.authorship_score or 85},
                                {'trait_type': 'Institution', 'value': 'Stylos University'},
                                {'trait_type': 'Course', 'value': 'Academic Writing'},
                                {'trait_type': 'Verification Date', 'value': record.verification_timestamp.strftime('%Y-%m-%d') if record.verification_timestamp else 'N/A'}
                            ]
                        },
                        'verification_proof': {
                            'authorshipScore': record.authorship_score or 85,
                            'aiProbability': max(0, 100 - (record.authorship_score or 85)),
                            'duplicateStatus': 'UNIQUE'
                        }
                    }
                    tokens.append(token_info)
                
                return tokens
            
        except Exception as e:
            logger.error(f"Failed to get demo portfolio tokens: {e}")
            return []

    def _load_contract_abi(self) -> Optional[List[Dict]]:
        """Load contract ABI from deployment artifacts."""
        try:
            import os
            import json
            
            # Try to load from exported ABI file
            abi_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'config',
                'contracts',
                'ProofOfAuthorship.json'
            )
            
            if os.path.exists(abi_path):
                with open(abi_path, 'r') as f:
                    contract_data = json.load(f)
                    logger.info(f"Loaded contract ABI from {abi_path}")
                    return contract_data.get('abi', [])
            else:
                logger.warning(f"ABI file not found at {abi_path}, using minimal ABI")
                # Return minimal ABI for basic functionality
                return [
                    {
                        "inputs": [
                            {"name": "_contentHash", "type": "bytes32"},
                            {"name": "_stylometricHash", "type": "bytes32"},
                            {"name": "_ipfsHash", "type": "string"},
                            {"name": "_student", "type": "address"},
                            {"name": "_authorshipScore", "type": "uint8"},
                            {"name": "_aiProbability", "type": "uint8"},
                            {"name": "_institutionId", "type": "string"},
                            {"name": "_courseId", "type": "string"}
                        ],
                        "name": "mintProofToken",
                        "outputs": [{"name": "", "type": "uint256"}],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    },
                    {
                        "inputs": [
                            {"name": "_tokenId", "type": "uint256"}
                        ],
                        "name": "getSubmission",
                        "outputs": [
                            {
                                "components": [
                                    {"name": "contentHash", "type": "bytes32"},
                                    {"name": "stylometricHash", "type": "bytes32"},
                                    {"name": "ipfsHash", "type": "string"},
                                    {"name": "student", "type": "address"},
                                    {"name": "timestamp", "type": "uint256"},
                                    {"name": "authorshipScore", "type": "uint8"},
                                    {"name": "aiProbability", "type": "uint8"},
                                    {"name": "verified", "type": "bool"},
                                    {"name": "institutionId", "type": "string"},
                                    {"name": "courseId", "type": "string"}
                                ],
                                "name": "",
                                "type": "tuple"
                            }
                        ],
                        "stateMutability": "view",
                        "type": "function"
                    },
                    {
                        "inputs": [
                            {"name": "_student", "type": "address"}
                        ],
                        "name": "getStudentSubmissions",
                        "outputs": [{"name": "", "type": "uint256[]"}],
                        "stateMutability": "view",
                        "type": "function"
                    },
                    {
                        "anonymous": False,
                        "inputs": [
                            {"indexed": True, "name": "tokenId", "type": "uint256"},
                            {"indexed": True, "name": "student", "type": "address"},
                            {"indexed": True, "name": "contentHash", "type": "bytes32"},
                            {"indexed": False, "name": "ipfsHash", "type": "string"},
                            {"indexed": False, "name": "authorshipScore", "type": "uint8"},
                            {"indexed": False, "name": "aiProbability", "type": "uint8"}
                        ],
                        "name": "SubmissionMinted",
                        "type": "event"
                    }
                ]
        except Exception as e:
            logger.error(f"Failed to load contract ABI: {e}")
            return None
    
    async def create_blockchain_attestation(
        self,
        submission_id: int,
        essay_content: str,
        verification_results: Dict[str, Any],
        student_address: str,
        institution_id: str = "stylos-university",
        course_id: str = "default"
    ) -> BlockchainRecord:
        """
        Create blockchain attestation for verified submission.
        
        Args:
            submission_id: Database submission ID
            essay_content: Full essay text
            verification_results: Verification results from ML/DL pipeline
            student_address: Student's blockchain address
            institution_id: Institution identifier
            course_id: Course identifier
            
        Returns:
            BlockchainRecord instance
        """
        try:
            # Create blockchain record
            # Get network configuration
            network_id = getattr(settings, 'NETWORK_ID', 31337)
            network_name = getattr(settings, 'NETWORK_NAME', 'localhost')
            
            blockchain_record = BlockchainRecord(
                submission_id=submission_id,
                contract_address=self.contract_address,
                content_hash=self._generate_content_hash(essay_content),
                authorship_score=int(verification_results.get('authorship_score', 0)),
                verification_timestamp=datetime.utcnow(),
                status=BlockchainStatus.PENDING,
                network_id=network_id,
                network_name=network_name
            )
            
            # Store essay and metadata on IPFS
            metadata = {
                'verification_results': verification_results,
                'submission_id': submission_id,
                'student_address': student_address,
                'institution_id': institution_id,
                'course_id': course_id
            }
            
            content_hash, metadata_hash = await ipfs_service.store_essay(
                essay_content,
                metadata,
                str(submission_id)
            )
            
            blockchain_record.ipfs_hash = content_hash
            blockchain_record.ipfs_metadata_hash = metadata_hash
            
            # Submit transaction to blockchain
            if self.contract and self.account:
                tx_hash = await self._submit_mint_transaction(
                    blockchain_record,
                    student_address,
                    institution_id,
                    course_id
                )
                blockchain_record.transaction_hash = tx_hash
                blockchain_record.status = BlockchainStatus.SUBMITTED
                blockchain_record.submitted_at = datetime.utcnow()
            
            logger.info(f"Blockchain attestation created for submission {submission_id}")
            return blockchain_record
            
        except Exception as e:
            logger.error(f"Failed to create blockchain attestation: {e}")
            blockchain_record.status = BlockchainStatus.FAILED
            blockchain_record.error_message = str(e)
            return blockchain_record
    
    async def _submit_mint_transaction(
        self,
        record: BlockchainRecord,
        student_address: str,
        institution_id: str,
        course_id: str
    ) -> str:
        """Submit mint transaction to smart contract."""
        try:
            # Prepare transaction data
            content_hash_bytes = bytes.fromhex(record.content_hash)
            stylometric_hash = self._generate_stylometric_hash(record.submission_id)
            
            # Get chain ID from settings
            chain_id = getattr(settings, 'CHAIN_ID', 31337)
            
            # Build transaction
            transaction = self.contract.functions.mintProofToken(
                content_hash_bytes,
                stylometric_hash,
                record.ipfs_hash,
                Web3.to_checksum_address(student_address),
                record.authorship_score,
                100 - record.authorship_score,  # AI probability inverse
                institution_id,
                course_id
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': chain_id
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Transaction submitted: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to submit mint transaction: {e}")
            raise
    
    async def check_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """
        Check the status of a blockchain transaction.
        
        Args:
            tx_hash: Transaction hash to check
            
        Returns:
            Transaction status information
        """
        try:
            # Get transaction receipt
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            if receipt:
                status = {
                    'hash': tx_hash,
                    'status': 'CONFIRMED' if receipt['status'] == 1 else 'FAILED',
                    'block_number': receipt['blockNumber'],
                    'block_hash': receipt['blockHash'].hex(),
                    'gas_used': receipt['gasUsed'],
                    'confirmations': self.w3.eth.block_number - receipt['blockNumber']
                }
                
                # Extract token ID from logs if successful
                if receipt['status'] == 1 and receipt['logs']:
                    for log in receipt['logs']:
                        try:
                            decoded_log = self.contract.events.SubmissionMinted().processLog(log)
                            status['token_id'] = str(decoded_log['args']['tokenId'])
                            break
                        except Exception:
                            continue
                
                return status
            else:
                return {
                    'hash': tx_hash,
                    'status': 'PENDING',
                    'confirmations': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to check transaction status: {e}")
            return {
                'hash': tx_hash,
                'status': 'UNKNOWN',
                'error': str(e)
            }
    
    async def update_blockchain_record_status(self, record: BlockchainRecord) -> BlockchainRecord:
        """Update blockchain record with latest transaction status."""
        if not record.transaction_hash:
            return record
        
        try:
            status_info = await self.check_transaction_status(record.transaction_hash)
            
            if status_info['status'] == 'CONFIRMED':
                record.status = BlockchainStatus.CONFIRMED
                record.confirmed_at = datetime.utcnow()
                record.block_number = status_info.get('block_number')
                record.block_hash = status_info.get('block_hash')
                record.gas_used = status_info.get('gas_used')
                record.token_id = status_info.get('token_id')
                
            elif status_info['status'] == 'FAILED':
                record.status = BlockchainStatus.FAILED
                record.error_message = "Transaction failed on blockchain"
                
            # Update retry logic
            if record.status == BlockchainStatus.FAILED and record.retry_count < record.max_retries:
                record.retry_count += 1
                record.status = BlockchainStatus.PENDING
                logger.info(f"Retrying blockchain transaction for submission {record.submission_id}")
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to update blockchain record status: {e}")
            return record
    

    async def get_student_portfolio_tokens(self, student_address: str) -> List[Dict[str, Any]]:
        """
        Get formatted tokens for portfolio display.
        
        Args:
            student_address: Student's blockchain address
            
        Returns:
            List of formatted token information for frontend
        """
        try:
            # Get tokens from database instead of blockchain for demo
            from core.database import AsyncSessionLocal
            from models.user import User
            from models.submission import Submission
            from models.blockchain import BlockchainRecord
            from sqlalchemy import select
            
            async with AsyncSessionLocal() as db:
                # Find user by wallet address
                user_stmt = select(User).filter(User.wallet_address == student_address)
                user_result = await db.execute(user_stmt)
                user = user_result.scalar_one_or_none()
                
                if not user:
                    return []
                
                # Get user's blockchain records
                stmt = select(BlockchainRecord, Submission).join(Submission).filter(
                    Submission.user_id == user.id,
                    BlockchainRecord.status == BlockchainStatus.CONFIRMED
                )
                result = await db.execute(stmt)
                records = result.all()
                
                tokens = []
                for record, submission in records:
                    token_info = {
                        'tokenId': record.token_id or str(1000 + record.id),
                        'submissionId': str(record.submission_id),
                        'submissionTitle': submission.title or f"Submission {submission.id}",
                        'mintedAt': record.confirmed_at.isoformat() if record.confirmed_at else record.created_at.isoformat(),
                        'transactionHash': record.transaction_hash,
                        'ipfsMetadata': {
                            'name': f"Proof of Authorship - {submission.title or f'Submission {submission.id}'}",
                            'description': f"Academic authorship verification for submission {submission.id}",
                            'attributes': [
                                {'trait_type': 'Authorship Score', 'value': record.authorship_score or 85},
                                {'trait_type': 'Institution', 'value': 'Stylos University'},
                                {'trait_type': 'Course', 'value': getattr(submission, 'course_id', 'Academic Writing')},
                                {'trait_type': 'Verification Date', 'value': record.verification_timestamp.strftime('%Y-%m-%d') if record.verification_timestamp else 'N/A'}
                            ]
                        },
                        'verificationProof': {
                            'authorshipScore': record.authorship_score or 85,
                            'aiProbability': max(0, 100 - (record.authorship_score or 85)),
                            'duplicateStatus': 'UNIQUE'
                        }
                    }
                    tokens.append(token_info)
                
                return tokens
            
        except Exception as e:
            logger.error(f"Failed to get student portfolio tokens: {e}")
            return []

    async def get_student_tokens(self, student_address: str) -> List[Dict[str, Any]]:
        """
        Get all tokens owned by a student.
        
        Args:
            student_address: Student's blockchain address
            
        Returns:
            List of token information
        """
        try:
            if not self.contract:
                return []
            
            # Get student submissions from contract
            token_ids = self.contract.functions.getStudentSubmissions(
                Web3.to_checksum_address(student_address)
            ).call()
            
            tokens = []
            for token_id in token_ids:
                submission_data = self.contract.functions.getSubmission(token_id).call()
                
                token_info = {
                    'token_id': str(token_id),
                    'content_hash': submission_data[0].hex(),
                    'ipfs_hash': submission_data[2],
                    'timestamp': datetime.fromtimestamp(submission_data[4]),
                    'authorship_score': submission_data[5],
                    'ai_probability': submission_data[6],
                    'verified': submission_data[7],
                    'institution_id': submission_data[8],
                    'course_id': submission_data[9]
                }
                tokens.append(token_info)
            
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to get student tokens: {e}")
            return []
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics."""
        try:
            network_name = getattr(settings, 'NETWORK_NAME', 'localhost')
            chain_id = getattr(settings, 'CHAIN_ID', 31337)
            
            latest_block = self.w3.eth.get_block('latest')
            gas_price = self.w3.eth.gas_price
            
            # Estimate confirmation time based on network
            confirmation_time = 1 if network_name == 'localhost' else 2
            
            stats = {
                'network': network_name,
                'chain_id': chain_id,
                'block_height': latest_block['number'],
                'gas_price': Web3.from_wei(gas_price, 'gwei'),
                'network_status': 'HEALTHY',
                'estimated_confirmation_time': confirmation_time,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get network stats: {e}")
            network_name = getattr(settings, 'NETWORK_NAME', 'localhost')
            return {
                'network': network_name,
                'chain_id': getattr(settings, 'CHAIN_ID', 31337),
                'block_height': 0,
                'gas_price': '20',
                'network_status': 'OFFLINE',
                'estimated_confirmation_time': 1,
                'last_updated': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_stylometric_hash(self, submission_id: int) -> bytes:
        """Generate stylometric signature hash."""
        # In production, this would use actual stylometric features
        signature = f"stylometric_signature_{submission_id}"
        return hashlib.sha256(signature.encode()).digest()
    
    async def retry_failed_attestation(self, record: BlockchainRecord) -> BlockchainRecord:
        """Retry a failed blockchain attestation."""
        if record.retry_count >= record.max_retries:
            logger.warning(f"Max retries exceeded for submission {record.submission_id}")
            return record
        
        try:
            record.retry_count += 1
            record.status = BlockchainStatus.PENDING
            record.error_message = None
            
            # Retry the transaction with higher gas price
            if self.contract and self.account:
                # Implementation would retry the mint transaction
                pass
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to retry attestation: {e}")
            record.status = BlockchainStatus.FAILED
            record.error_message = str(e)
            return record


# Global blockchain service instance
blockchain_service = BlockchainService()
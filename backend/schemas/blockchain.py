"""
Blockchain-related Pydantic schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from models.blockchain import BlockchainStatus


class BlockchainRecordBase(BaseModel):
    """Base blockchain record schema."""
    submission_id: int
    contract_address: str
    content_hash: str
    authorship_score: Optional[int] = None
    network_id: int = 137
    network_name: str = "polygon"


class BlockchainRecordCreate(BlockchainRecordBase):
    """Schema for creating blockchain record."""
    pass


class BlockchainRecordResponse(BlockchainRecordBase):
    """Schema for blockchain record response."""
    id: int
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    token_id: Optional[str] = None
    ipfs_hash: Optional[str] = None
    ipfs_metadata_hash: Optional[str] = None
    status: BlockchainStatus
    gas_used: Optional[int] = None
    gas_price: Optional[str] = None
    verification_timestamp: datetime
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    
    # Computed properties
    is_confirmed: bool = Field(default=False)
    explorer_url: str = Field(default="")
    ipfs_url: str = Field(default="")
    
    class Config:
        from_attributes = True


class SoulboundTokenMetadata(BaseModel):
    """Metadata for Soulbound Token."""
    name: str
    description: str
    attributes: List[Dict[str, Any]]


class SoulboundTokenResponse(BaseModel):
    """Schema for Soulbound Token response."""
    tokenId: str = Field(alias="token_id")
    submissionId: str = Field(alias="submission_id")
    submissionTitle: str = Field(alias="submission_title")
    mintedAt: datetime = Field(alias="minted_at")
    transactionHash: str = Field(alias="transaction_hash")
    ipfsMetadata: SoulboundTokenMetadata = Field(alias="ipfs_metadata")
    verificationProof: Dict[str, Any] = Field(alias="verification_proof")
    
    class Config:
        populate_by_name = True


class PortfolioValue(BaseModel):
    """Portfolio value metrics."""
    academicCredibility: int = Field(ge=0, le=100, alias="academic_credibility")
    uniquenessScore: int = Field(ge=0, le=100, alias="uniqueness_score")
    consistencyRating: int = Field(ge=0, le=100, alias="consistency_rating")
    
    class Config:
        populate_by_name = True


class RecentActivity(BaseModel):
    """Recent portfolio activity."""
    type: str = Field(pattern="^(MINT|VERIFY|UPDATE)$")
    timestamp: datetime
    description: str
    transaction_hash: Optional[str] = None


class BlockchainPortfolioResponse(BaseModel):
    """Schema for blockchain portfolio response."""
    total_tokens: int
    total_verified_submissions: int
    portfolio_value: PortfolioValue
    tokens: List[SoulboundTokenResponse]
    recent_activity: List[RecentActivity]


class TransactionStatusResponse(BaseModel):
    """Schema for transaction status response."""
    hash: str
    status: str = Field(pattern="^(PENDING|CONFIRMED|FAILED|UNKNOWN)$")
    confirmations: int = 0
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None


class NetworkStatsResponse(BaseModel):
    """Schema for network statistics response."""
    network: str
    chain_id: Optional[int] = None
    block_height: int
    gas_price: str
    network_status: str = Field(pattern="^(HEALTHY|CONGESTED|OFFLINE)$")
    estimated_confirmation_time: int
    last_updated: str


class IPFSContentResponse(BaseModel):
    """Schema for IPFS content response."""
    ipfs_hash: str
    content: Any  # Can be string or dict depending on content type
    content_type: str = Field(pattern="^(essay|metadata)$")
    retrieved_at: str


class IPFSUploadRequest(BaseModel):
    """Schema for IPFS upload request."""
    content: str
    filename: Optional[str] = None
    encrypt: bool = True
    pin: bool = True


class IPFSUploadResponse(BaseModel):
    """Schema for IPFS upload response."""
    ipfs_hash: str
    size: int
    encrypted: bool
    pinned: bool
    uploaded_at: str


class BlockchainAttestationRequest(BaseModel):
    """Schema for blockchain attestation request."""
    submission_id: int
    student_address: str
    institution_id: Optional[str] = "stylos-university"
    course_id: Optional[str] = "default"


class BlockchainAttestationResponse(BaseModel):
    """Schema for blockchain attestation response."""
    message: str
    transaction_hash: Optional[str] = None
    ipfs_hash: Optional[str] = None
    status: BlockchainStatus


class RetryAttestationResponse(BaseModel):
    """Schema for retry attestation response."""
    message: str
    transaction_hash: Optional[str] = None
    retry_count: int


class BlockchainRecordsListResponse(BaseModel):
    """Schema for paginated blockchain records list."""
    records: List[BlockchainRecordResponse]
    total: int
    page: int
    total_pages: int


class IPFSStorageStats(BaseModel):
    """Schema for IPFS storage statistics."""
    total_pinned_items: int
    pinned_content: List[Dict[str, Any]]
    redundancy_nodes: int
    encryption_enabled: bool
    last_updated: str


class ContractEventLog(BaseModel):
    """Schema for smart contract event log."""
    event_name: str
    block_number: int
    transaction_hash: str
    timestamp: datetime
    args: Dict[str, Any]


class ContractMetrics(BaseModel):
    """Schema for smart contract metrics."""
    contract_address: str
    total_supply: str
    is_paused: bool
    current_block: int
    recent_activity: Dict[str, Any]


class GovernanceAction(BaseModel):
    """Schema for governance actions."""
    action_type: str = Field(pattern="^(GRANT_ROLE|REVOKE_ROLE|PAUSE|UNPAUSE|UPGRADE)$")
    target_address: Optional[str] = None
    role_name: Optional[str] = None
    executed_by: str
    executed_at: datetime
    transaction_hash: str


class EmergencyResponse(BaseModel):
    """Schema for emergency response."""
    action: str
    status: str
    message: str
    transaction_hash: Optional[str] = None
    executed_at: datetime
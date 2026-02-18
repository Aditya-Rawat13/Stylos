"""
Blockchain endpoints for IPFS and smart contract integration.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from datetime import datetime
import logging
import asyncio

from core.database import get_db
from core.auth import get_current_user
from models.user import User
from models.submission import Submission
from models.blockchain import BlockchainRecord, BlockchainStatus
from services.blockchain_service import blockchain_service
from services.ipfs_service import ipfs_service
from schemas.blockchain import (
    BlockchainRecordResponse,
    BlockchainPortfolioResponse,
    TransactionStatusResponse,
    NetworkStatsResponse,
    IPFSContentResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/records", response_model=List[BlockchainRecordResponse])
async def get_blockchain_records(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[BlockchainStatus] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get blockchain records for the current user."""
    try:
        # Query blockchain records for user's submissions
        query = select(BlockchainRecord).join(Submission).filter(
            Submission.user_id == current_user.id
        )
        
        if status:
            query = query.filter(BlockchainRecord.status == status)
        
        # Get total count
        from sqlalchemy import func
        count_query = select(func.count()).select_from(BlockchainRecord).join(Submission).filter(
            Submission.user_id == current_user.id
        )
        if status:
            count_query = count_query.filter(BlockchainRecord.status == status)
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        # Pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        result = await db.execute(query)
        records = result.scalars().all()
        
        return {
            "records": records,
            "total": total,
            "page": page,
            "total_pages": (total + limit - 1) // limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get blockchain records: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve blockchain records")


@router.get("/health")
async def blockchain_health():
    """Health check for blockchain service."""
    return {"status": "ok", "service": "blockchain"}


@router.get("/portfolio")
async def get_blockchain_portfolio(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's blockchain portfolio with tokens and statistics."""
    try:
        from sqlalchemy import select
        
        # Get user's blockchain records using async select
        stmt = select(BlockchainRecord).join(Submission).filter(
            Submission.user_id == current_user.id,
            BlockchainRecord.status == BlockchainStatus.CONFIRMED
        )
        result = await db.execute(stmt)
        records = result.scalars().all()
        
        # Get tokens from blockchain if user has wallet address
        tokens = []
        if hasattr(current_user, 'wallet_address') and current_user.wallet_address:
            try:
                tokens = await blockchain_service.get_student_portfolio_tokens(current_user.wallet_address)
            except Exception:
                # Fallback to demo tokens if blockchain is offline
                tokens = await blockchain_service.get_demo_portfolio_tokens(current_user.wallet_address)
        
        # Calculate portfolio statistics
        total_tokens = len(tokens)
        total_verified = len([r for r in records if r.token_id])
        
        portfolio_value = {
            "academicCredibility": min(95, 60 + (total_verified * 5)),
            "uniquenessScore": min(100, 70 + (total_verified * 3)),
            "consistencyRating": min(100, 65 + (total_verified * 4))
        }
        
        # Recent activity
        recent_activity = []
        for record in records[-5:]:  # Last 5 records
            if record.confirmed_at:
                recent_activity.append({
                    "type": "MINT",
                    "timestamp": record.confirmed_at.isoformat(),
                    "description": f"Proof-of-authorship token minted for submission",
                    "transaction_hash": record.transaction_hash
                })
        
        return {
            "total_tokens": total_tokens,
            "total_verified_submissions": total_verified,
            "portfolio_value": portfolio_value,
            "tokens": tokens,
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Failed to get blockchain portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve blockchain portfolio")


@router.get("/records/{submission_id}", response_model=BlockchainRecordResponse)
async def get_blockchain_record(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get blockchain record for a specific submission."""
    try:
        # Verify user owns the submission
        stmt = select(Submission).filter(
            Submission.id == submission_id,
            Submission.user_id == current_user.id
        )
        result = await db.execute(stmt)
        submission = result.scalar_one_or_none()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Get blockchain record
        stmt = select(BlockchainRecord).filter(
            BlockchainRecord.submission_id == submission_id
        )
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(status_code=404, detail="Blockchain record not found")
        
        return record
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get blockchain record: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve blockchain record")


@router.post("/{submission_id}/attest")
async def create_blockchain_attestation(
    submission_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create blockchain attestation for verified submission."""
    try:
        # Verify user owns the submission
        stmt = select(Submission).filter(
            Submission.id == submission_id,
            Submission.user_id == current_user.id
        )
        result = await db.execute(stmt)
        submission = result.scalar_one_or_none()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Check if submission is verified
        if not submission.verification_result or submission.verification_result.get('overall_status') != 'PASS':
            raise HTTPException(status_code=400, detail="Submission must be verified before blockchain attestation")
        
        # Check if blockchain record already exists
        stmt = select(BlockchainRecord).filter(
            BlockchainRecord.submission_id == submission_id
        )
        result = await db.execute(stmt)
        existing_record = result.scalar_one_or_none()
        
        if existing_record:
            raise HTTPException(status_code=400, detail="Blockchain attestation already exists")
        
        # Get user's wallet address
        wallet_address = getattr(current_user, 'wallet_address', None)
        if not wallet_address:
            raise HTTPException(status_code=400, detail="User wallet address not configured")
        
        # Create blockchain attestation
        blockchain_record = await blockchain_service.create_blockchain_attestation(
            submission_id=submission_id,
            essay_content=submission.content,
            verification_results=submission.verification_result,
            student_address=wallet_address,
            institution_id=getattr(current_user, 'institution_id', 'stylos-university'),
            course_id=getattr(submission, 'course_id', 'default')
        )
        
        # Save to database
        db.add(blockchain_record)
        await db.commit()
        
        # Schedule background task to monitor transaction
        background_tasks.add_task(monitor_blockchain_transaction, blockchain_record.id)
        
        return {
            "message": "Blockchain attestation created",
            "transaction_hash": blockchain_record.transaction_hash,
            "ipfs_hash": blockchain_record.ipfs_hash,
            "status": blockchain_record.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create blockchain attestation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create blockchain attestation")


@router.get("/status/{transaction_hash}", response_model=TransactionStatusResponse)
async def get_transaction_status(
    transaction_hash: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a blockchain transaction."""
    try:
        status_info = await blockchain_service.check_transaction_status(transaction_hash)
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get transaction status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get transaction status")


@router.post("/retry/{submission_id}")
async def retry_blockchain_attestation(
    submission_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retry failed blockchain attestation."""
    try:
        # Get blockchain record
        stmt = select(BlockchainRecord).filter(
            BlockchainRecord.submission_id == submission_id
        )
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(status_code=404, detail="Blockchain record not found")
        
        # Verify user owns the submission
        stmt = select(Submission).filter(
            Submission.id == submission_id,
            Submission.user_id == current_user.id
        )
        result = await db.execute(stmt)
        submission = result.scalar_one_or_none()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Retry the attestation
        updated_record = await blockchain_service.retry_failed_attestation(record)
        
        # Update database
        await db.merge(updated_record)
        await db.commit()
        
        # Schedule monitoring
        background_tasks.add_task(monitor_blockchain_transaction, record.id)
        
        return {
            "message": "Blockchain attestation retry initiated",
            "transaction_hash": updated_record.transaction_hash,
            "retry_count": updated_record.retry_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry blockchain attestation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry blockchain attestation")


@router.get("/network-stats", response_model=NetworkStatsResponse)
async def get_network_stats():
    """Get current blockchain network statistics."""
    try:
        stats = await blockchain_service.get_network_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get network stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get network statistics")


@router.get("/ipfs/{ipfs_hash}", response_model=IPFSContentResponse)
async def get_ipfs_content(
    ipfs_hash: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get content from IPFS (if user has access)."""
    try:
        # Verify user has access to this IPFS content
        stmt = select(BlockchainRecord).join(Submission).filter(
            BlockchainRecord.ipfs_hash == ipfs_hash,
            Submission.user_id == current_user.id
        )
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(status_code=403, detail="Access denied to IPFS content")
        
        # Retrieve content from IPFS
        if ipfs_hash == record.ipfs_hash:
            # Essay content
            content = await ipfs_service.retrieve_essay(ipfs_hash)
            content_type = "essay"
        elif ipfs_hash == record.ipfs_metadata_hash:
            # Metadata
            content = await ipfs_service.retrieve_metadata(ipfs_hash)
            content_type = "metadata"
        else:
            raise HTTPException(status_code=404, detail="IPFS content not found")
        
        return {
            "ipfs_hash": ipfs_hash,
            "content": content,
            "content_type": content_type,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get IPFS content: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve IPFS content")


@router.get("/export")
async def export_portfolio(
    export_format: str = Query("json", regex="^(json|pdf)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export user's blockchain portfolio."""
    try:
        # Get user's blockchain records
        stmt = select(BlockchainRecord).join(Submission).filter(
            Submission.user_id == current_user.id
        )
        result = await db.execute(stmt)
        records = result.scalars().all()
        
        if export_format == "json":
            portfolio_data = {
                "user_id": current_user.id,
                "exported_at": datetime.utcnow().isoformat(),
                "records": [
                    {
                        "submission_id": r.submission_id,
                        "transaction_hash": r.transaction_hash,
                        "token_id": r.token_id,
                        "ipfs_hash": r.ipfs_hash,
                        "status": r.status,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    }
                    for r in records
                ]
            }
            
            return portfolio_data
        
        # PDF export would be implemented here
        raise HTTPException(status_code=501, detail="PDF export not yet implemented")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to export portfolio")


@router.get("/tokens/{token_id}")
async def get_token_details(
    token_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get details for a specific soulbound token."""
    try:
        # Find blockchain record by token ID
        stmt = select(BlockchainRecord).join(Submission).filter(
            BlockchainRecord.token_id == token_id,
            Submission.user_id == current_user.id
        )
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(status_code=404, detail="Token not found")
        
        # Get submission details
        stmt = select(Submission).filter(Submission.id == record.submission_id)
        result = await db.execute(stmt)
        submission = result.scalar_one_or_none()
        
        return {
            "tokenId": record.token_id,
            "submissionId": record.submission_id,
            "transactionHash": record.transaction_hash,
            "ipfsHash": record.ipfs_hash,
            "status": record.status,
            "metadata": {
                "title": submission.title if submission else "Unknown",
                "verificationScore": submission.confidence_score if submission else 0,
                "timestamp": record.created_at.isoformat() if record.created_at else None
            },
            "createdAt": record.created_at.isoformat() if record.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get token details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get token details")


@router.get("/verify/{transaction_hash}")
async def verify_transaction(
    transaction_hash: str,
    db: AsyncSession = Depends(get_db)
):
    """Verify a blockchain transaction (public endpoint)."""
    try:
        # Find blockchain record
        stmt = select(BlockchainRecord).filter(
            BlockchainRecord.transaction_hash == transaction_hash
        )
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        
        if not record:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Get submission details (limited info for public access)
        stmt = select(Submission).filter(Submission.id == record.submission_id)
        result = await db.execute(stmt)
        submission = result.scalar_one_or_none()
        
        return {
            "transactionHash": record.transaction_hash,
            "status": record.status,
            "tokenId": record.token_id,
            "ipfsHash": record.ipfs_hash,
            "verified": record.status == BlockchainStatus.CONFIRMED,
            "timestamp": record.created_at.isoformat() if record.created_at else None,
            "metadata": {
                "hasVerification": bool(submission and submission.verification_result),
                "confidenceScore": submission.confidence_score if submission else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify transaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify transaction")


@router.get("/stream/{transaction_hash}")
async def stream_transaction_updates(
    transaction_hash: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Stream real-time updates for a blockchain transaction (Server-Sent Events)."""
    from fastapi.responses import StreamingResponse
    import json
    
    async def event_generator():
        """Generate SSE events for transaction updates."""
        try:
            # Verify transaction belongs to user
            stmt = select(BlockchainRecord).join(Submission).filter(
                BlockchainRecord.transaction_hash == transaction_hash,
                Submission.user_id == current_user.id
            )
            result = await db.execute(stmt)
            record = result.scalar_one_or_none()
            
            if not record:
                yield f"data: {json.dumps({'error': 'Transaction not found'})}\n\n"
                return
            
            # Send initial status
            yield f"data: {json.dumps({'status': record.status, 'confirmations': 0})}\n\n"
            
            # For confirmed/failed transactions, close the stream
            if record.status in [BlockchainStatus.CONFIRMED, BlockchainStatus.FAILED]:
                yield f"data: {json.dumps({'status': record.status, 'done': True})}\n\n"
                return
            
            # Poll for updates
            for i in range(60):  # Poll for 60 seconds
                await asyncio.sleep(1)
                
                # Refresh record from database
                db.refresh(record)
                
                confirmations = min(i, 12)  # Simulate confirmations
                yield f"data: {json.dumps({'status': record.status, 'confirmations': confirmations})}\n\n"
                
                if record.status in [BlockchainStatus.CONFIRMED, BlockchainStatus.FAILED]:
                    yield f"data: {json.dumps({'status': record.status, 'confirmations': 12, 'done': True})}\n\n"
                    break
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def monitor_blockchain_transaction(record_id: int):
    """Background task to monitor blockchain transaction status."""
    from core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            stmt = select(BlockchainRecord).filter(BlockchainRecord.id == record_id)
            result = await db.execute(stmt)
            record = result.scalar_one_or_none()
            
            if not record or not record.transaction_hash:
                return
            
            # Monitor for up to 10 minutes
            for _ in range(30):  # 30 attempts, 20 seconds apart
                updated_record = await blockchain_service.update_blockchain_record_status(record)
                
                # Update database
                await db.merge(updated_record)
                await db.commit()
                
                # Stop monitoring if confirmed or failed
                if updated_record.status in [BlockchainStatus.CONFIRMED, BlockchainStatus.FAILED]:
                    break
                
                # Wait before next check
                await asyncio.sleep(20)
                
        except Exception as e:
            logger.error(f"Error monitoring blockchain transaction: {e}")
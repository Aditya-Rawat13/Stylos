"""
Service to keep writing profile counts synchronized with actual submissions
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from models.user import WritingProfile
from models.submission import Submission
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ProfileSyncService:
    """Service to synchronize writing profile counts with actual submissions"""
    
    @staticmethod
    async def sync_profile_counts(db: AsyncSession, user_id: int):
        """Synchronize writing profile counts for a specific user"""
        try:
            # Get the user's writing profile
            profile_result = await db.execute(
                select(WritingProfile).where(WritingProfile.user_id == user_id)
            )
            profile = profile_result.scalar_one_or_none()
            
            if not profile:
                logger.warning(f"No writing profile found for user {user_id}")
                return
            
            # Count actual submissions
            total_subs_result = await db.execute(
                select(func.count(Submission.id)).where(Submission.user_id == user_id)
            )
            actual_submissions = total_subs_result.scalar()
            
            # Calculate total words
            total_words_result = await db.execute(
                select(func.coalesce(func.sum(func.length(Submission.content)), 0))
                .where(Submission.user_id == user_id)
            )
            actual_words = total_words_result.scalar() or 0
            
            # Update if counts don't match
            if (profile.total_submissions != actual_submissions or 
                profile.total_words != actual_words):
                
                await db.execute(
                    update(WritingProfile)
                    .where(WritingProfile.user_id == user_id)
                    .values(
                        total_submissions=actual_submissions,
                        total_words=actual_words,
                        last_updated=datetime.now(timezone.utc)
                    )
                )
                
                logger.info(f"Synced profile for user {user_id}: {actual_submissions} submissions, {actual_words} words")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error syncing profile counts for user {user_id}: {e}")
            raise
    
    @staticmethod
    async def sync_all_profiles(db: AsyncSession):
        """Synchronize all writing profile counts"""
        try:
            # Get all writing profiles
            profiles_result = await db.execute(select(WritingProfile))
            profiles = profiles_result.scalars().all()
            
            synced_count = 0
            
            for profile in profiles:
                if await ProfileSyncService.sync_profile_counts(db, profile.user_id):
                    synced_count += 1
            
            if synced_count > 0:
                await db.commit()
                logger.info(f"Synced {synced_count} writing profiles")
            
            return synced_count
            
        except Exception as e:
            logger.error(f"Error syncing all profiles: {e}")
            raise

# Global instance
profile_sync_service = ProfileSyncService()

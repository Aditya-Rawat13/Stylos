#!/usr/bin/env python3
"""
Initialize database with all tables and sample data.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.ext.asyncio import AsyncSession
from core.database import engine, Base, AsyncSessionLocal
from models.user import User, WritingProfile, UserRole
from models.submission import Submission, SubmissionStatus
from models.verification import VerificationResult, VerificationStatus
from models.blockchain import BlockchainRecord, BlockchainStatus
from services.auth_service import auth_service
from core.security import security


async def init_database():
    """Initialize database with all tables."""
    print("🔧 Initializing database...")
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)  # Fresh start
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ Database tables created successfully")
        
        # Create sample data
        await create_sample_data()
        
        print("🎉 Database initialization complete!")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise


async def create_sample_data():
    """Create sample users and data for testing."""
    print("📝 Creating sample data...")
    
    async with AsyncSessionLocal() as db:
        try:
            # Create test user
            test_user = User(
                email="test@stylos.dev",
                hashed_password=security.get_password_hash("TestPassword123"),
                full_name="Test User",
                role=UserRole.STUDENT,
                is_active=True,
                is_verified=True,
                institution_id="UNIV001",
                student_id="STU001"
            )
            db.add(test_user)
            await db.commit()
            await db.refresh(test_user)
            
            # Create admin user
            admin_user = User(
                email="admin@stylos.dev",
                hashed_password=security.get_password_hash("AdminPassword123"),
                full_name="Admin User",
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True
            )
            db.add(admin_user)
            await db.commit()
            await db.refresh(admin_user)
            
            # Create writing profile for test user
            writing_profile = WritingProfile(
                user_id=test_user.id,
                lexical_features={
                    "type_token_ratio": 0.75,
                    "avg_word_length": 5.2,
                    "vocabulary_richness": 0.68,
                    "total_words": 1500,
                    "unique_words": 450,
                    "total_sentences": 85
                },
                syntactic_features={
                    "avg_sentence_length": 18.5,
                    "sentence_length_variance": 12.3,
                    "punctuation_frequency": {
                        "comma": 0.045,
                        "period": 0.032,
                        "semicolon": 0.008
                    }
                },
                semantic_features={
                    "function_word_frequency": {
                        "articles": 0.12,
                        "prepositions": 0.15,
                        "pronouns": 0.08
                    },
                    "text_length": 8500,
                    "word_count": 1500
                },
                total_submissions=3,
                total_words=4500,
                avg_confidence_score=85,
                is_initialized=True,
                sample_essays="1,2,3"
            )
            db.add(writing_profile)
            await db.commit()
            
            # Create sample submission
            sample_submission = Submission(
                user_id=test_user.id,
                filename="sample_essay.txt",
                file_path="uploads/sample_essay.txt",
                file_size=2048,
                file_hash="abc123def456",
                title="Sample Academic Essay",
                content="This is a sample academic essay for testing purposes. It demonstrates the writing style and capabilities of the student. The essay covers various topics and showcases different writing techniques.",
                word_count=150,
                status=SubmissionStatus.VERIFIED,
                assignment_id="ASSIGN001",
                assignment_title="Introduction to Academic Writing",
                course_id="COURSE001"
            )
            db.add(sample_submission)
            await db.commit()
            await db.refresh(sample_submission)
            
            # Create verification result
            verification = VerificationResult(
                submission_id=sample_submission.id,
                status=VerificationStatus.COMPLETED,
                authorship_score=0.87,
                authorship_confidence=0.92,
                is_authentic=True,
                similarity_score=0.12,
                has_duplicates=False,
                ai_probability=0.08,
                is_ai_generated=False,
                ai_detection_model="enhanced_detector_v2",
                stylometric_analysis={
                    "lexical": {
                        "ttr": 0.78,
                        "avg_word_length": 5.1,
                        "vocabulary_richness": 0.71
                    },
                    "syntactic": {
                        "avg_sentence_length": 19.2,
                        "complexity_score": 0.74
                    },
                    "semantic": {
                        "coherence_score": 0.83,
                        "topic_consistency": 0.79
                    }
                },
                semantic_analysis={
                    "topics": ["academic writing", "education", "analysis"],
                    "sentiment": "neutral",
                    "formality": 0.85
                },
                processing_time_seconds=3.2,
                model_versions={
                    "authorship": "v2.1",
                    "ai_detection": "v1.8",
                    "similarity": "v1.5"
                }
            )
            db.add(verification)
            await db.commit()
            
            # Create blockchain record
            blockchain_record = BlockchainRecord(
                submission_id=sample_submission.id,
                contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",
                content_hash="abc123def456",
                authorship_score=87,
                verification_timestamp=sample_submission.created_at,
                status=BlockchainStatus.CONFIRMED,
                transaction_hash="0x1234567890abcdef",
                block_number=12345,
                token_id="1",
                ipfs_hash="QmTest123",
                network_name="polygon"
            )
            db.add(blockchain_record)
            await db.commit()
            
            print("✅ Sample data created successfully")
            print(f"   - Test user: test@stylos.dev (password: TestPassword123)")
            print(f"   - Admin user: admin@stylos.dev (password: AdminPassword123)")
            print(f"   - Sample submission with full verification and blockchain record")
            
        except Exception as e:
            await db.rollback()
            print(f"❌ Error creating sample data: {e}")
            raise


async def verify_database():
    """Verify database setup by checking tables and data."""
    print("🔍 Verifying database setup...")
    
    async with AsyncSessionLocal() as db:
        try:
            # Check users
            from sqlalchemy import select, func
            
            user_count = await db.execute(select(func.count(User.id)))
            user_total = user_count.scalar()
            
            profile_count = await db.execute(select(func.count(WritingProfile.id)))
            profile_total = profile_count.scalar()
            
            submission_count = await db.execute(select(func.count(Submission.id)))
            submission_total = submission_count.scalar()
            
            verification_count = await db.execute(select(func.count(VerificationResult.id)))
            verification_total = verification_count.scalar()
            
            blockchain_count = await db.execute(select(func.count(BlockchainRecord.id)))
            blockchain_total = blockchain_count.scalar()
            
            print(f"📊 Database Statistics:")
            print(f"   - Users: {user_total}")
            print(f"   - Writing Profiles: {profile_total}")
            print(f"   - Submissions: {submission_total}")
            print(f"   - Verification Results: {verification_total}")
            print(f"   - Blockchain Records: {blockchain_total}")
            
            if all([user_total > 0, profile_total > 0, submission_total > 0, verification_total > 0, blockchain_total > 0]):
                print("✅ Database verification successful - all models working!")
            else:
                print("⚠️  Some tables are empty - this might be expected for a fresh install")
                
        except Exception as e:
            print(f"❌ Error verifying database: {e}")
            raise


if __name__ == "__main__":
    print("🚀 Starting database initialization...")
    asyncio.run(init_database())
    asyncio.run(verify_database())
    print("🎯 Database setup complete!")
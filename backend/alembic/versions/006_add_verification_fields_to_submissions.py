"""Add verification fields to submissions table

Revision ID: 006
Revises: 005
Create Date: 2024-11-24

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    """Add verification result fields to submissions table."""
    # Add new columns for storing verification results
    op.add_column('submissions', sa.Column('ai_detection_score', sa.JSON(), nullable=True))
    op.add_column('submissions', sa.Column('similarity_score', sa.JSON(), nullable=True))
    op.add_column('submissions', sa.Column('confidence_score', sa.JSON(), nullable=True))
    op.add_column('submissions', sa.Column('stylometric_features', sa.JSON(), nullable=True))
    op.add_column('submissions', sa.Column('verification_summary', sa.JSON(), nullable=True))


def downgrade():
    """Remove verification result fields from submissions table."""
    op.drop_column('submissions', 'verification_summary')
    op.drop_column('submissions', 'stylometric_features')
    op.drop_column('submissions', 'confidence_score')
    op.drop_column('submissions', 'similarity_score')
    op.drop_column('submissions', 'ai_detection_score')

"""
Database constraint to prevent 0.0 authorship scores
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    # Add check constraint to ensure authorship_score is never 0.0
    op.execute("""
        ALTER TABLE verification_results 
        ADD CONSTRAINT check_authorship_score_not_zero 
        CHECK (authorship_score > 0.0 AND authorship_score <= 1.0)
    """)

def downgrade():
    op.execute("ALTER TABLE verification_results DROP CONSTRAINT check_authorship_score_not_zero")

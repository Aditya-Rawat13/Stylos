"""Add security and compliance tables

Revision ID: 004_add_security_tables
Revises: 003_add_blockchain_tables
Create Date: 2024-11-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '004_add_security_tables'
down_revision = '003_add_blockchain_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Add security and compliance tables."""
    
    # Encryption Keys table
    op.create_table('encryption_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key_id', sa.String(length=100), nullable=False),
        sa.Column('key_type', sa.String(length=50), nullable=False),
        sa.Column('algorithm', sa.String(length=50), nullable=False, default='AES-256-GCM'),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rotated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rotation_reason', sa.String(length=200), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_encryption_keys_id'), 'encryption_keys', ['id'], unique=False)
    op.create_index(op.f('ix_encryption_keys_key_id'), 'encryption_keys', ['key_id'], unique=True)
    
    # Security Incidents table
    op.create_table('security_incidents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('incident_type', sa.String(length=100), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, default='OPEN'),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('affected_user_id', sa.Integer(), nullable=True),
        sa.Column('source_ip', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('detection_method', sa.String(length=100), nullable=False),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.Integer(), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_security_incidents_id'), 'security_incidents', ['id'], unique=False)
    op.create_index(op.f('ix_security_incidents_incident_type'), 'security_incidents', ['incident_type'], unique=False)
    op.create_index(op.f('ix_security_incidents_severity'), 'security_incidents', ['severity'], unique=False)
    op.create_index(op.f('ix_security_incidents_created_at'), 'security_incidents', ['created_at'], unique=False)
    
    # Threat Intelligence table
    op.create_table('threat_intelligence',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('indicator_type', sa.String(length=50), nullable=False),
        sa.Column('indicator_value', sa.String(length=500), nullable=False),
        sa.Column('threat_type', sa.String(length=100), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('source', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_threat_intelligence_id'), 'threat_intelligence', ['id'], unique=False)
    op.create_index(op.f('ix_threat_intelligence_indicator_type'), 'threat_intelligence', ['indicator_type'], unique=False)
    op.create_index(op.f('ix_threat_intelligence_indicator_value'), 'threat_intelligence', ['indicator_value'], unique=False)
    
    # Data Retention Policies table
    op.create_table('data_retention_policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('policy_name', sa.String(length=100), nullable=False),
        sa.Column('data_type', sa.String(length=100), nullable=False),
        sa.Column('retention_period_days', sa.Integer(), nullable=False),
        sa.Column('deletion_method', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('legal_basis', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('policy_name')
    )
    op.create_index(op.f('ix_data_retention_policies_id'), 'data_retention_policies', ['id'], unique=False)
    op.create_index(op.f('ix_data_retention_policies_data_type'), 'data_retention_policies', ['data_type'], unique=False)
    
    # Data Processing Records table (GDPR Article 30)
    op.create_table('data_processing_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('processing_activity', sa.String(length=200), nullable=False),
        sa.Column('data_controller', sa.String(length=200), nullable=False),
        sa.Column('data_processor', sa.String(length=200), nullable=True),
        sa.Column('data_categories', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('data_subjects', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('purposes', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('legal_basis', sa.String(length=200), nullable=False),
        sa.Column('recipients', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('third_country_transfers', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('retention_period', sa.String(length=200), nullable=False),
        sa.Column('security_measures', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_processing_records_id'), 'data_processing_records', ['id'], unique=False)
    
    # Consent Records table
    op.create_table('consent_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('consent_type', sa.String(length=100), nullable=False),
        sa.Column('purpose', sa.String(length=200), nullable=False),
        sa.Column('legal_basis', sa.String(length=100), nullable=False),
        sa.Column('consent_given', sa.Boolean(), nullable=False),
        sa.Column('consent_method', sa.String(length=100), nullable=False),
        sa.Column('consent_text', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('withdrawn_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('withdrawal_method', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_consent_records_id'), 'consent_records', ['id'], unique=False)
    op.create_index(op.f('ix_consent_records_user_id'), 'consent_records', ['user_id'], unique=False)
    
    # Data Subject Requests table
    op.create_table('data_subject_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('request_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, default='PENDING'),
        sa.Column('request_details', sa.Text(), nullable=True),
        sa.Column('verification_method', sa.String(length=100), nullable=True),
        sa.Column('verification_completed', sa.Boolean(), nullable=False, default=False),
        sa.Column('response_data', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deadline', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_subject_requests_id'), 'data_subject_requests', ['id'], unique=False)
    op.create_index(op.f('ix_data_subject_requests_user_id'), 'data_subject_requests', ['user_id'], unique=False)
    op.create_index(op.f('ix_data_subject_requests_request_type'), 'data_subject_requests', ['request_type'], unique=False)


def downgrade():
    """Remove security and compliance tables."""
    
    # Drop tables in reverse order
    op.drop_table('data_subject_requests')
    op.drop_table('consent_records')
    op.drop_table('data_processing_records')
    op.drop_table('data_retention_policies')
    op.drop_table('threat_intelligence')
    op.drop_table('security_incidents')
    op.drop_table('encryption_keys')
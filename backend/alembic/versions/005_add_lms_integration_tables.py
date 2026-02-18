"""Add LMS integration tables

Revision ID: 005_add_lms_integration_tables
Revises: 004_add_security_tables
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_add_lms_integration_tables'
down_revision = '004_add_security_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Create lms_configurations table
    op.create_table('lms_configurations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('institution_id', sa.String(length=255), nullable=False),
        sa.Column('lms_type', sa.String(length=50), nullable=False),
        sa.Column('base_url', sa.String(length=500), nullable=False),
        sa.Column('api_key_encrypted', sa.Text(), nullable=True),
        sa.Column('client_id', sa.String(length=255), nullable=True),
        sa.Column('client_secret_encrypted', sa.Text(), nullable=True),
        sa.Column('webhook_secret_encrypted', sa.Text(), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=True),
        sa.Column('sso_enabled', sa.Boolean(), nullable=True),
        sa.Column('saml_metadata_url', sa.String(length=500), nullable=True),
        sa.Column('saml_entity_id', sa.String(length=255), nullable=True),
        sa.Column('oauth_authorize_url', sa.String(length=500), nullable=True),
        sa.Column('oauth_token_url', sa.String(length=500), nullable=True),
        sa.Column('lti_consumer_key', sa.String(length=255), nullable=True),
        sa.Column('lti_shared_secret_encrypted', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_configurations_id'), 'lms_configurations', ['id'], unique=False)
    op.create_index(op.f('ix_lms_configurations_institution_id'), 'lms_configurations', ['institution_id'], unique=True)

    # Create lms_courses table
    op.create_table('lms_courses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('lms_config_id', sa.Integer(), nullable=False),
        sa.Column('lms_course_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=500), nullable=False),
        sa.Column('code', sa.String(length=100), nullable=True),
        sa.Column('term', sa.String(length=100), nullable=True),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('enrollment_term_id', sa.String(length=255), nullable=True),
        sa.Column('sis_course_id', sa.String(length=255), nullable=True),
        sa.Column('workflow_state', sa.String(length=50), nullable=True),
        sa.Column('last_synced', sa.DateTime(), nullable=True),
        sa.Column('sync_enabled', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lms_config_id'], ['lms_configurations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_courses_id'), 'lms_courses', ['id'], unique=False)

    # Create lms_assignments table
    op.create_table('lms_assignments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('lms_assignment_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('due_date', sa.DateTime(), nullable=True),
        sa.Column('unlock_date', sa.DateTime(), nullable=True),
        sa.Column('lock_date', sa.DateTime(), nullable=True),
        sa.Column('points_possible', sa.Float(), nullable=True),
        sa.Column('submission_types', sa.JSON(), nullable=True),
        sa.Column('allowed_extensions', sa.JSON(), nullable=True),
        sa.Column('published', sa.Boolean(), nullable=True),
        sa.Column('workflow_state', sa.String(length=50), nullable=True),
        sa.Column('auto_verification_enabled', sa.Boolean(), nullable=True),
        sa.Column('grade_passback_enabled', sa.Boolean(), nullable=True),
        sa.Column('last_synced', sa.DateTime(), nullable=True),
        sa.Column('sync_enabled', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['course_id'], ['lms_courses.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_assignments_id'), 'lms_assignments', ['id'], unique=False)

    # Create lms_submission_records table
    op.create_table('lms_submission_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('assignment_id', sa.Integer(), nullable=False),
        sa.Column('submission_id', sa.Integer(), nullable=True),
        sa.Column('lms_submission_id', sa.String(length=255), nullable=False),
        sa.Column('lms_user_id', sa.String(length=255), nullable=False),
        sa.Column('submitted_at', sa.DateTime(), nullable=True),
        sa.Column('grade', sa.Float(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('workflow_state', sa.String(length=50), nullable=True),
        sa.Column('submission_type', sa.String(length=50), nullable=True),
        sa.Column('body', sa.Text(), nullable=True),
        sa.Column('url', sa.String(length=500), nullable=True),
        sa.Column('attachments', sa.JSON(), nullable=True),
        sa.Column('grade_submitted', sa.Boolean(), nullable=True),
        sa.Column('grade_submitted_at', sa.DateTime(), nullable=True),
        sa.Column('grade_submission_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['assignment_id'], ['lms_assignments.id'], ),
        sa.ForeignKeyConstraint(['submission_id'], ['submissions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_submission_records_id'), 'lms_submission_records', ['id'], unique=False)

    # Create lms_webhook_subscriptions table
    op.create_table('lms_webhook_subscriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('lms_config_id', sa.Integer(), nullable=False),
        sa.Column('lms_webhook_id', sa.String(length=255), nullable=True),
        sa.Column('webhook_url', sa.String(length=500), nullable=False),
        sa.Column('events', sa.JSON(), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=True),
        sa.Column('last_delivery', sa.DateTime(), nullable=True),
        sa.Column('total_deliveries', sa.Integer(), nullable=True),
        sa.Column('failed_deliveries', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lms_config_id'], ['lms_configurations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_webhook_subscriptions_id'), 'lms_webhook_subscriptions', ['id'], unique=False)

    # Create lms_webhook_deliveries table
    op.create_table('lms_webhook_deliveries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=False),
        sa.Column('event_id', sa.String(length=255), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('response_code', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('delivered_at', sa.DateTime(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('event_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['subscription_id'], ['lms_webhook_subscriptions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_webhook_deliveries_id'), 'lms_webhook_deliveries', ['id'], unique=False)

    # Create lms_user_mappings table
    op.create_table('lms_user_mappings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('lms_config_id', sa.Integer(), nullable=False),
        sa.Column('lms_user_id', sa.String(length=255), nullable=False),
        sa.Column('lms_login_id', sa.String(length=255), nullable=True),
        sa.Column('sis_user_id', sa.String(length=255), nullable=True),
        sa.Column('lms_name', sa.String(length=255), nullable=True),
        sa.Column('lms_email', sa.String(length=255), nullable=True),
        sa.Column('lms_roles', sa.JSON(), nullable=True),
        sa.Column('last_synced', sa.DateTime(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lms_config_id'], ['lms_configurations.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_user_mappings_id'), 'lms_user_mappings', ['id'], unique=False)

    # Create lms_sync_logs table
    op.create_table('lms_sync_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('lms_config_id', sa.Integer(), nullable=False),
        sa.Column('sync_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('items_processed', sa.Integer(), nullable=True),
        sa.Column('items_created', sa.Integer(), nullable=True),
        sa.Column('items_updated', sa.Integer(), nullable=True),
        sa.Column('items_failed', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['lms_config_id'], ['lms_configurations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_sync_logs_id'), 'lms_sync_logs', ['id'], unique=False)

    # Create lms_grade_passbacks table
    op.create_table('lms_grade_passbacks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('submission_id', sa.Integer(), nullable=False),
        sa.Column('lms_assignment_id', sa.String(length=255), nullable=False),
        sa.Column('lms_user_id', sa.String(length=255), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=True),
        sa.Column('last_attempt', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('lms_response', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['submission_id'], ['submissions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lms_grade_passbacks_id'), 'lms_grade_passbacks', ['id'], unique=False)

    # Add LMS-related columns to existing users table
    op.add_column('users', sa.Column('lms_user_id', sa.String(length=255), nullable=True))
    op.add_column('users', sa.Column('lms_context_id', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_users_lms_user_id'), 'users', ['lms_user_id'], unique=False)

    # Add LMS-related columns to existing submissions table
    op.add_column('submissions', sa.Column('lms_assignment_id', sa.String(length=255), nullable=True))
    op.add_column('submissions', sa.Column('lms_submission_id', sa.String(length=255), nullable=True))
    op.add_column('submissions', sa.Column('lms_grade_submitted', sa.Boolean(), nullable=True))
    op.add_column('submissions', sa.Column('lms_grade_submitted_at', sa.DateTime(), nullable=True))
    op.create_index(op.f('ix_submissions_lms_assignment_id'), 'submissions', ['lms_assignment_id'], unique=False)


def downgrade():
    # Remove indexes and columns from existing tables
    op.drop_index(op.f('ix_submissions_lms_assignment_id'), table_name='submissions')
    op.drop_column('submissions', 'lms_grade_submitted_at')
    op.drop_column('submissions', 'lms_grade_submitted')
    op.drop_column('submissions', 'lms_submission_id')
    op.drop_column('submissions', 'lms_assignment_id')
    
    op.drop_index(op.f('ix_users_lms_user_id'), table_name='users')
    op.drop_column('users', 'lms_context_id')
    op.drop_column('users', 'lms_user_id')

    # Drop LMS integration tables
    op.drop_table('lms_grade_passbacks')
    op.drop_table('lms_sync_logs')
    op.drop_table('lms_user_mappings')
    op.drop_table('lms_webhook_deliveries')
    op.drop_table('lms_webhook_subscriptions')
    op.drop_table('lms_submission_records')
    op.drop_table('lms_assignments')
    op.drop_table('lms_courses')
    op.drop_table('lms_configurations')
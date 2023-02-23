"""empty message

Revision ID: 04ba34c0cd80
Revises: 
Create Date: 2023-02-22 22:57:00.644118

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '04ba34c0cd80'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('prediction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('temperature', sa.Integer(), nullable=True),
    sa.Column('humidity', sa.Integer(), nullable=True),
    sa.Column('sound_volume', sa.Integer(), nullable=True),
    sa.Column('anomaly_score', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('prediction')
    # ### end Alembic commands ###

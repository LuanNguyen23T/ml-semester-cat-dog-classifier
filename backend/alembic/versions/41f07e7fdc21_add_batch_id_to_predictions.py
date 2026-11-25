"""Add batch_id to predictions

Revision ID: 41f07e7fdc21
Revises: 9c6dba231b24
Create Date: 2026-09-29
"""
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "41f07e7fdc21"
down_revision: Union[str, None] = "9c6dba231b24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index("ix_predictions_batch_id", "predictions", ["batch_id"], unique=False)

    # Existing all-model requests share one uploaded image across distinct methods.
    connection = op.get_bind()
    image_rows = connection.execute(sa.text("""
        SELECT image_key
        FROM predictions
        WHERE batch_id IS NULL
        GROUP BY image_key
        HAVING COUNT(DISTINCT method) > 1
    """)).fetchall()
    for row in image_rows:
        connection.execute(
            sa.text("""
                UPDATE predictions
                SET batch_id = :batch_id
                WHERE image_key = :image_key AND batch_id IS NULL
            """),
            {"batch_id": uuid.uuid4(), "image_key": row.image_key},
        )


def downgrade() -> None:
    op.drop_index("ix_predictions_batch_id", table_name="predictions")
    op.drop_column("predictions", "batch_id")

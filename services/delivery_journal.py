"""Durable delivery receipts. An uncertain network outcome is never auto-retried."""
from __future__ import annotations

import json
from typing import Optional

from storage.agent2_database import agent2_connection, initialize_agent2_database


class UncertainDeliveryError(RuntimeError):
    pass


def claim_part(operation: str, part: int, payload_hash: str, *, database_path=None):
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM delivery_parts WHERE operation = ? AND part = ?",
            (operation, part),
        ).fetchone()
        if row:
            if row["payload_hash"] != payload_hash:
                raise ValueError("Delivery content changed after a send attempt. Start a new delivery.")
            if row["status"] == "sent":
                return json.loads(row["receipt_json"])
            if row["status"] == "pending":
                raise UncertainDeliveryError(
                    f"Delivery part {part + 1} may already have arrived. Check the recipient "
                    "before resolving the uncertain delivery; it will not be sent twice automatically."
                )
        connection.execute(
            """INSERT INTO delivery_parts(operation, part, payload_hash, status)
               VALUES (?, ?, ?, 'pending') ON CONFLICT(operation, part)
               DO UPDATE SET status = 'pending', receipt_json = NULL""",
            (operation, part, payload_hash),
        )
    return None


def finish_part(operation: str, part: int, receipt: Optional[dict], *, database_path=None):
    with agent2_connection(database_path) as connection:
        connection.execute(
            """UPDATE delivery_parts SET status = ?, receipt_json = ?
               WHERE operation = ? AND part = ?""",
            ("sent" if receipt is not None else "failed",
             json.dumps(receipt) if receipt is not None else None, operation, part),
        )


def pending_parts(operation: str, *, database_path=None) -> list[int]:
    initialize_agent2_database(database_path)
    with agent2_connection(database_path) as connection:
        return [row["part"] for row in connection.execute(
            "SELECT part FROM delivery_parts WHERE operation = ? AND status = 'pending' ORDER BY part",
            (operation,),
        ).fetchall()]


def resolve_part(operation: str, part: int, received: bool, *, database_path=None):
    """Only call after the user explicitly checks the recipient's messages."""
    with agent2_connection(database_path) as connection:
        connection.execute(
            """UPDATE delivery_parts SET status = ?, receipt_json = ?
               WHERE operation = ? AND part = ? AND status = 'pending'""",
            ("sent" if received else "failed",
             json.dumps({"confirmed_by_user": True}) if received else None, operation, part),
        )

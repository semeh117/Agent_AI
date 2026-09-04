"""Persistent LangGraph checkpoints stored in Agent 2's SQLite database.

The project's pinned LangGraph version predates the current official SQLite
package. This focused saver implements that installed checkpoint interface
using parameterized SQL and LangGraph's safe default msgpack serializer. It
can be replaced by the official saver when the LangGraph stack is upgraded.
"""

from __future__ import annotations

from pathlib import Path
from random import random
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from storage.agent2_database import (
    agent2_connection,
    get_agent2_database_path,
    initialize_agent2_database,
)


class Agent2SqliteSaver(BaseCheckpointSaver[str]):
    """SQLite-backed checkpointer compatible with the installed LangGraph."""

    def __init__(self, database_path: Optional[str | Path] = None) -> None:
        super().__init__()
        self.database_path = get_agent2_database_path(database_path)
        initialize_agent2_database(self.database_path)

    @staticmethod
    def _config(
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        with agent2_connection(self.database_path) as connection:
            for channel, version in versions.items():
                row = connection.execute(
                    """
                    SELECT value_type, value
                    FROM graph_blobs
                    WHERE thread_id = ? AND checkpoint_ns = ?
                      AND channel = ? AND version = ?
                    """,
                    (thread_id, checkpoint_ns, channel, str(version)),
                ).fetchone()
                if row is not None and row["value_type"] != "empty":
                    values[channel] = self.serde.loads_typed(
                        (row["value_type"], bytes(row["value"]))
                    )
        return values

    def _tuple_from_row(self, row: Any) -> CheckpointTuple:
        thread_id = row["thread_id"]
        checkpoint_ns = row["checkpoint_ns"]
        checkpoint_id = row["checkpoint_id"]
        checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], bytes(row["checkpoint"]))
        )
        metadata = self.serde.loads_typed(
            (row["metadata_type"], bytes(row["metadata"]))
        )
        with agent2_connection(self.database_path) as connection:
            write_rows = connection.execute(
                """
                SELECT task_id, channel, value_type, value
                FROM graph_writes
                WHERE thread_id = ? AND checkpoint_ns = ?
                  AND checkpoint_id = ?
                ORDER BY task_id, write_index
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()

        parent_id = row["parent_checkpoint_id"]
        return CheckpointTuple(
            config=self._config(thread_id, checkpoint_ns, checkpoint_id),
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id,
                    checkpoint_ns,
                    checkpoint["channel_versions"],
                ),
            },
            metadata=metadata,
            parent_config=(
                self._config(thread_id, checkpoint_ns, parent_id)
                if parent_id
                else None
            ),
            pending_writes=[
                (
                    write["task_id"],
                    write["channel"],
                    self.serde.loads_typed(
                        (write["value_type"], bytes(write["value"]))
                    ),
                )
                for write in write_rows
            ],
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)
        with agent2_connection(self.database_path) as connection:
            if checkpoint_id:
                row = connection.execute(
                    """
                    SELECT * FROM graph_checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                      AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM graph_checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()
        return self._tuple_from_row(row) if row is not None else None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if config is not None:
            configurable = config["configurable"]
            clauses.append("thread_id = ?")
            parameters.append(str(configurable["thread_id"]))
            if "checkpoint_ns" in configurable:
                clauses.append("checkpoint_ns = ?")
                parameters.append(str(configurable["checkpoint_ns"]))
            if checkpoint_id := get_checkpoint_id(config):
                clauses.append("checkpoint_id = ?")
                parameters.append(checkpoint_id)
        if before and (before_id := get_checkpoint_id(before)):
            clauses.append("checkpoint_id < ?")
            parameters.append(before_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM graph_checkpoints {where} ORDER BY checkpoint_id DESC"
        with agent2_connection(self.database_path) as connection:
            rows = connection.execute(sql, parameters).fetchall()

        yielded = 0
        for row in rows:
            checkpoint_tuple = self._tuple_from_row(row)
            if filter and not all(
                checkpoint_tuple.metadata.get(key) == value
                for key, value in filter.items()
            ):
                continue
            if limit is not None and yielded >= limit:
                break
            yielded += 1
            yield checkpoint_tuple

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_copy = checkpoint.copy()
        channel_values = checkpoint_copy.pop("channel_values")
        checkpoint_type, checkpoint_bytes = self.serde.dumps_typed(checkpoint_copy)
        metadata_type, metadata_bytes = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )

        with agent2_connection(self.database_path) as connection:
            for channel, version in new_versions.items():
                if channel in channel_values:
                    value_type, value = self.serde.dumps_typed(
                        channel_values[channel]
                    )
                else:
                    value_type, value = "empty", b""
                connection.execute(
                    """
                    INSERT OR REPLACE INTO graph_blobs(
                        thread_id, checkpoint_ns, channel, version,
                        value_type, value
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        channel,
                        str(version),
                        value_type,
                        value,
                    ),
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO graph_checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id,
                    checkpoint_type, checkpoint, metadata_type, metadata,
                    parent_checkpoint_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_type,
                    checkpoint_bytes,
                    metadata_type,
                    metadata_bytes,
                    configurable.get("checkpoint_id"),
                ),
            )
        return self._config(thread_id, checkpoint_ns, checkpoint["id"])

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable["checkpoint_id"])
        with agent2_connection(self.database_path) as connection:
            for position, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, position)
                value_type, value_bytes = self.serde.dumps_typed(value)
                command = (
                    "INSERT OR REPLACE" if write_index < 0 else "INSERT OR IGNORE"
                )
                connection.execute(
                    f"""
                    {command} INTO graph_writes(
                        thread_id, checkpoint_ns, checkpoint_id, task_id,
                        write_index, channel, value_type, value, task_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_index,
                        channel,
                        value_type,
                        value_bytes,
                        task_path,
                    ),
                )

    def delete_thread(self, thread_id: str) -> None:
        with agent2_connection(self.database_path) as connection:
            for table in ("graph_writes", "graph_blobs", "graph_checkpoints"):
                connection.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?",
                    (thread_id,),
                )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_number = 0
        elif isinstance(current, int):
            current_number = current
        else:
            current_number = int(current.split(".")[0])
        next_number = current_number + 1
        return f"{next_number:032}.{random():016}"

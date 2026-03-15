"""
Redis Infrastructure
- MessageQueue: task queue using Redis Streams
- DeadLetterQueue: DLQ using a dedicated Redis List
- CacheStore: general-purpose key-value cache
"""
from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis
import structlog

from config.settings import get_settings
from src.domain.repositories.interfaces import CacheRepository, MessageQueueRepository

log = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Redis client factory
# ---------------------------------------------------------------------------

def create_redis_client(db: int = 0) -> aioredis.Redis:
    return aioredis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password or None,
        db=db,
        max_connections=settings.redis.max_connections,
        decode_responses=False,  # We handle encoding ourselves
    )


# ---------------------------------------------------------------------------
# Message Queue (Redis Streams)
# ---------------------------------------------------------------------------

class RedisMessageQueueRepository(MessageQueueRepository):
    """
    Uses Redis Streams for reliable message delivery.
    - Main queue: ingestion stream
    - DLQ: separate stream with error metadata
    - Consumer group for distributed workers
    """

    CONSUMER_GROUP = "pipeline-workers"
    CONSUMER_NAME = "worker-{id}"
    DLQ_STREAM = "stream:dlq"

    def __init__(self, client: aioredis.Redis, dlq_client: aioredis.Redis) -> None:
        self._client = client
        self._dlq_client = dlq_client

    @classmethod
    def create(cls) -> "RedisMessageQueueRepository":
        return cls(
            client=create_redis_client(db=settings.redis.db_queue),
            dlq_client=create_redis_client(db=settings.redis.db_dlq),
        )

    def _stream_name(self, queue: str) -> str:
        return f"stream:{queue}"

    async def _ensure_consumer_group(self, stream: str) -> None:
        try:
            await self._client.xgroup_create(
                stream, self.CONSUMER_GROUP, id="0", mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def enqueue(
        self, queue: str, message: dict, priority: int = 0
    ) -> str:
        stream = self._stream_name(queue)
        await self._ensure_consumer_group(stream)

        fields = {
            "payload": json.dumps(message),
            "priority": str(priority),
            "enqueued_at": str(time.time()),
        }
        msg_id = await self._client.xadd(stream, fields)
        log.debug("queue.enqueued", queue=queue, msg_id=msg_id)
        return msg_id.decode() if isinstance(msg_id, bytes) else msg_id

    async def dequeue(
        self, queue: str, timeout: int = 30
    ) -> dict | None:
        stream = self._stream_name(queue)
        await self._ensure_consumer_group(stream)

        import socket
        consumer = f"worker-{socket.gethostname()}"

        results = await self._client.xreadgroup(
            groupname=self.CONSUMER_GROUP,
            consumername=consumer,
            streams={stream: ">"},
            count=1,
            block=timeout * 1000,
        )

        if not results:
            return None

        _, messages = results[0]
        if not messages:
            return None

        msg_id, fields = messages[0]
        payload = json.loads(fields[b"payload"])
        payload["__stream_id"] = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        payload["__stream"] = stream
        return payload

    async def ack(self, queue: str, stream_id: str) -> None:
        stream = self._stream_name(queue)
        await self._client.xack(stream, self.CONSUMER_GROUP, stream_id)

    async def enqueue_dlq(self, message: dict, reason: str) -> None:
        fields = {
            "payload": json.dumps(message),
            "reason": reason,
            "failed_at": str(time.time()),
        }
        await self._dlq_client.xadd(self.DLQ_STREAM, fields)
        log.warning(
            "dlq.enqueued",
            document_id=message.get("document_id"),
            reason=reason,
        )

    async def list_dlq(self, limit: int = 50) -> list[dict]:
        results = await self._dlq_client.xrange(
            self.DLQ_STREAM, count=limit
        )
        items = []
        for msg_id, fields in results:
            item = {
                "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                "payload": json.loads(fields[b"payload"]),
                "reason": fields[b"reason"].decode(),
                "failed_at": float(fields[b"failed_at"]),
            }
            items.append(item)
        return items

    async def requeue_from_dlq(self, message_id: str) -> bool:
        """Move a DLQ message back to the main ingestion queue."""
        results = await self._dlq_client.xrange(
            self.DLQ_STREAM,
            min=message_id,
            max=message_id,
            count=1,
        )
        if not results:
            return False

        _, fields = results[0]
        payload = json.loads(fields[b"payload"])

        # Reset retry count for manual requeue
        payload["retry_count"] = 0

        await self.enqueue(settings.worker.queue_name, payload)
        await self._dlq_client.xdel(self.DLQ_STREAM, message_id)

        log.info("dlq.requeued", message_id=message_id)
        return True

    async def dlq_count(self) -> int:
        info = await self._dlq_client.xlen(self.DLQ_STREAM)
        return info


# ---------------------------------------------------------------------------
# Cache (Redis key-value)
# ---------------------------------------------------------------------------

class RedisCacheRepository(CacheRepository):
    """
    General-purpose Redis cache.
    Used for: embedding cache, classification cache, dedup fingerprints.
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    @classmethod
    def create(cls) -> "RedisCacheRepository":
        return cls(create_redis_client(db=settings.redis.db_cache))

    async def get(self, key: str) -> bytes | None:
        value = await self._client.get(key)
        return value  # bytes or None

    async def set(
        self, key: str, value: bytes, ttl: int | None = None
    ) -> None:
        effective_ttl = ttl if ttl is not None else settings.redis.ttl_seconds
        await self._client.set(key, value, ex=effective_ttl)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))

    async def set_json(
        self, key: str, value: Any, ttl: int | None = None
    ) -> None:
        await self.set(key, json.dumps(value).encode(), ttl=ttl)

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw.decode())

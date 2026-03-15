"""
MinIO Object Storage Repository
Raw file storage with bucket lifecycle management.
"""
from __future__ import annotations

import io

import structlog
from miniopy_async import Minio

from config.settings import get_settings
from src.domain.repositories.interfaces import ObjectStorageRepository

log = structlog.get_logger(__name__)
settings = get_settings()


class MinioObjectStorageRepository(ObjectStorageRepository):

    def __init__(self, client: Minio) -> None:
        self._client = client

    @classmethod
    def create(cls) -> "MinioObjectStorageRepository":
        client = Minio(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            secure=settings.minio.secure,
        )
        return cls(client)

    async def ensure_buckets(self) -> None:
        for bucket in (
            settings.minio.bucket_raw,
            settings.minio.bucket_processed,
            settings.minio.bucket_dlq,
        ):
            exists = await self._client.bucket_exists(bucket)
            if not exists:
                await self._client.make_bucket(bucket)
                log.info("minio.bucket.created", bucket=bucket)

    async def upload(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict | None = None,
    ) -> str:
        stream = io.BytesIO(data)
        await self._client.put_object(
            bucket_name=bucket,
            object_name=key,
            data=stream,
            length=len(data),
            content_type=content_type,
            metadata=metadata or {},
        )
        log.debug("minio.upload.done", bucket=bucket, key=key, size=len(data))
        return key

    async def download(self, bucket: str, key: str) -> bytes:
        response = await self._client.get_object(bucket, key)
        data = await response.read()
        await response.close()
        return data

    async def delete(self, bucket: str, key: str) -> None:
        await self._client.remove_object(bucket, key)
        log.debug("minio.delete.done", bucket=bucket, key=key)

    async def presigned_url(
        self, bucket: str, key: str, expiry: int = 3600
    ) -> str:
        from datetime import timedelta
        url = await self._client.presigned_get_object(
            bucket, key, expires=timedelta(seconds=expiry)
        )
        return url

    async def move(
        self,
        src_bucket: str,
        src_key: str,
        dst_bucket: str,
        dst_key: str,
    ) -> None:
        from miniopy_async import CopySource
        await self._client.copy_object(
            dst_bucket,
            dst_key,
            CopySource(src_bucket, src_key),
        )
        await self.delete(src_bucket, src_key)
        log.debug(
            "minio.move.done",
            src=f"{src_bucket}/{src_key}",
            dst=f"{dst_bucket}/{dst_key}",
        )

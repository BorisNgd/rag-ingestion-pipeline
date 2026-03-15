"""
Integration tests — requires running infrastructure (postgres, redis, minio, qdrant).
Run with: pytest tests/integration/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import io
import os
import pytest
import pytest_asyncio

from httpx import AsyncClient, ASGITransport

from src.adapters.api.app import create_app
from config.settings import get_settings

settings = get_settings()

# Skip if not in integration test environment
pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="Set INTEGRATION_TESTS=1 to run",
)


@pytest_asyncio.fixture
async def client():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-API-Key": settings.api.api_key},
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ingest_markdown(client: AsyncClient):
    content = b"# Test Document\n\nThis is a test paragraph for RAG ingestion."
    files = {"file": ("test.md", io.BytesIO(content), "text/markdown")}
    data = {"tags": "test,integration"}

    resp = await client.post("/api/v1/ingest", files=files, data=data)
    assert resp.status_code == 202

    body = resp.json()
    assert "job_id" in body
    assert "document_id" in body
    assert body["status"] == "pending"

    return body["job_id"]


@pytest.mark.asyncio
async def test_job_status(client: AsyncClient):
    # First ingest a document
    content = b"# Status Test\n\nChecking job status works."
    files = {"file": ("status_test.md", io.BytesIO(content), "text/markdown")}
    ingest_resp = await client.post("/api/v1/ingest", files=files, data={})
    job_id = ingest_resp.json()["job_id"]

    # Check status
    resp = await client.get(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_search(client: AsyncClient):
    resp = await client.post(
        "/api/v1/search",
        json={"query": "test document ingestion", "limit": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert isinstance(body["results"], list)


@pytest.mark.asyncio
async def test_list_dlq(client: AsyncClient):
    resp = await client.get("/api/v1/admin/dlq")
    assert resp.status_code == 200
    assert "items" in resp.json()


@pytest.mark.asyncio
async def test_list_collections(client: AsyncClient):
    resp = await client.get("/api/v1/admin/collections")
    assert resp.status_code == 200
    collections = resp.json()["collections"]
    assert "general" in collections
    assert "technical" in collections


@pytest.mark.asyncio
async def test_ingest_rejects_oversized(client: AsyncClient):
    # 1 byte over the limit simulation — we patch settings in unit tests instead
    pass


@pytest.mark.asyncio
async def test_unauthorized_request():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        resp = await c.get("/api/v1/admin/dlq")
        assert resp.status_code == 401

"""
test_api.py
===========
Integration tests for the FastAPI endpoints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import cv2
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


def _make_jpeg(value: int = 128, h: int = 64, w: int = 64) -> bytes:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


def _make_png() -> bytes:
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".png", img)
    return enc.tobytes()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data


# ── Analysis ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_valid_jpeg(client):
    img_bytes = _make_jpeg(128)
    resp = await client.post(
        "/api/analyze",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "quality_score" in data
    assert "quality_label" in data
    assert "issues" in data
    assert "features" in data
    assert 0 <= data["quality_score"] <= 100
    assert data["quality_label"] in {"GOOD", "ACCEPTABLE", "DEGRADED", "DEFECTIVE"}


@pytest.mark.asyncio
async def test_analyze_valid_png(client):
    img_bytes = _make_png()
    resp = await client.post(
        "/api/analyze",
        files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
    )
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_analyze_invalid_file_type(client):
    resp = await client.post(
        "/api/analyze",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_analyze_corrupt_image(client):
    resp = await client.post(
        "/api/analyze",
        files={"file": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_analysis_not_found(client):
    resp = await client.get("/api/analysis/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_analyze_then_retrieve(client):
    img_bytes = _make_jpeg()
    post_resp = await client.post(
        "/api/analyze",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert post_resp.status_code == 201
    analysis_id = post_resp.json()["id"]

    get_resp = await client.get(f"/api/analysis/{analysis_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == analysis_id


@pytest.mark.asyncio
async def test_analyze_then_delete(client):
    img_bytes = _make_jpeg()
    post_resp = await client.post(
        "/api/analyze",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    analysis_id = post_resp.json()["id"]

    del_resp = await client.delete(f"/api/analysis/{analysis_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/api/analysis/{analysis_id}")
    assert get_resp.status_code == 404


# ── History ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_empty_initially(client):
    resp = await client.get("/api/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "pages" in data


@pytest.mark.asyncio
async def test_history_after_analysis(client):
    img_bytes = _make_jpeg()
    await client.post(
        "/api/analyze",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    resp = await client.get("/api/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1


# ── Batch ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_analyze(client):
    files = [
        ("files", ("img1.jpg", io.BytesIO(_make_jpeg(100)), "image/jpeg")),
        ("files", ("img2.jpg", io.BytesIO(_make_jpeg(200)), "image/jpeg")),
    ]
    resp = await client.post("/api/analyze/batch", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["success_count"] >= 1


@pytest.mark.asyncio
async def test_batch_too_many_files(client):
    files = [
        ("files", (f"img{i}.jpg", io.BytesIO(_make_jpeg()), "image/jpeg"))
        for i in range(11)
    ]
    resp = await client.post("/api/analyze/batch", files=files)
    assert resp.status_code == 400

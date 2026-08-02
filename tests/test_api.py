import io

import docx
import pytest
from fastapi.testclient import TestClient

from api.auth import API_KEY
from api.main import app

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": API_KEY}


def test_health_requires_no_auth():
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "plagiarism_index_documents" in body
    assert body["ai_detector_mode"] in {"heuristic", "sklearn", "transformer"}


def test_plagiarism_endpoint_requires_api_key():
    response = client.post("/v1/detect/plagiarism", json={"text": "hello world"})
    assert response.status_code == 401


def test_plagiarism_endpoint_rejects_wrong_api_key():
    response = client.post(
        "/v1/detect/plagiarism", json={"text": "hello world"}, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


def test_plagiarism_endpoint_returns_expected_shape_with_valid_key():
    response = client.post(
        "/v1/detect/plagiarism",
        json={"text": "Machine learning is a branch of artificial intelligence."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"score", "matches", "sentence_matches"}
    assert isinstance(body["matches"], list)


def test_plagiarism_endpoint_rejects_empty_text():
    response = client.post("/v1/detect/plagiarism", json={"text": ""}, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_ai_text_endpoint_returns_score_and_label():
    response = client.post(
        "/v1/detect/ai-text",
        json={"text": "Artificial intelligence is transforming industries worldwide."},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 100.0
    assert body["label"] in {"AI", "Human"}


def test_plagiarism_file_endpoint_accepts_docx_upload():
    buffer = io.BytesIO()
    document = docx.Document()
    document.add_paragraph("This is an uploaded document for the file endpoint test.")
    document.save(buffer)
    buffer.seek(0)

    response = client.post(
        "/v1/detect/plagiarism/file",
        files={"file": ("upload.docx", buffer, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert "score" in response.json()


def test_plagiarism_file_endpoint_rejects_unsupported_type():
    response = client.post(
        "/v1/detect/plagiarism/file",
        files={"file": ("upload.exe", io.BytesIO(b"not text"), "application/octet-stream")},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 422

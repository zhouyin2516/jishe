import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from config import settings

client = TestClient(app)

# Override the settings to ensure USE_ECS_AGENCY is False
settings.USE_ECS_AGENCY = False
settings.INTERNAL_API_KEY = "test_secret_key"
settings.DEV_AK = "fake_ak"
settings.DEV_SK = "fake_sk"

headers = {"X-Internal-Token": "test_secret_key"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_internal_token_rejection():
    # Calling without header should fail
    response = client.get("/api/internal/storage/list_groups")
    assert response.status_code == 403

def test_generate_sts():
    payload = {
        "group_id": "test_group",
        "file_key": "model.zip",
        "action": "upload"
    }
    response = client.post("/api/internal/storage/generate_sts", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "access_key" in data
    assert "security_token" in data
    assert data["security_token"] == "mock-sts-security-token-for-local-development"

@patch('services.obs_service.get_obs_client')
def test_list_groups(mock_get_client):
    # Mock OBS Client and ListObjects response
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.body.commonPrefixes = [{'prefix': 'group A/'}, {'prefix': 'group B/'}]
    mock_client.listObjects.return_value = mock_resp
    
    mock_get_client.return_value.__enter__.return_value = mock_client

    response = client.get("/api/internal/storage/list_groups", headers=headers)
    assert response.status_code == 200
    assert response.json() == ["group A", "group B"]

@patch('services.obs_service.get_obs_client')
def test_list_models(mock_get_client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = 200
    
    content = MagicMock()
    content.key = "test_group/models/unet.zip"
    content.size = 1024000
    content.lastModified = "2026-04-19T12:00:00Z"
    
    mock_resp.body.contents = [content]
    mock_resp.body.is_truncated = False
    mock_client.listObjects.return_value = mock_resp
    
    mock_get_client.return_value.__enter__.return_value = mock_client

    response = client.get("/api/internal/storage/list_models?group_id=test_group", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["group_id"] == "test_group"
    assert len(data["models"]) == 1
    assert data["models"][0]["file_key"] == "unet.zip"

@patch('services.obs_service.get_obs_client')
def test_object_meta(mock_get_client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.header.get.side_effect = lambda key, default=None: "5000" if key == "content-length" else "\"some-etag\""
    mock_client.getObjectMetadata.return_value = mock_resp
    mock_get_client.return_value.__enter__.return_value = mock_client

    response = client.get("/api/internal/storage/object_meta?group_id=test_group&file_key=model.zip", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["content_length"] == 5000
    assert data["etag"] == "some-etag"

@patch('services.obs_service.get_obs_client')
def test_delete_object(mock_get_client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = 204
    mock_client.deleteObject.return_value = mock_resp
    mock_get_client.return_value.__enter__.return_value = mock_client

    payload = {"group_id": "test_group", "file_key": "model.zip"}
    response = client.post("/api/internal/storage/delete_object", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["success"] is True

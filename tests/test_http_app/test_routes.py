from unittest import mock

import pytest
from fastapi.testclient import TestClient

from imaginairy.http_app.app import app

client = TestClient(app)


@pytest.fixture(name="red_b64")
def _red_b64():
    return b"iVBORw0KGgoAAAANSUhEUgAAAgAAAAIAAQMAAADOtka5AAAABlBMVEX/AAD///9BHTQRAAAANklEQVR4nO3BAQEAAACCIP+vbkhAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8G4IAAAHSeInwAAAAAElFTkSuQmCC"


@pytest.fixture()
def mock_generate_image_b64(monkeypatch, red_b64):
    fake_generate = mock.MagicMock(return_value=red_b64)
    monkeypatch.setattr(
        "imaginairy.http_app.stablestudio.routes.generate_image_b64", fake_generate
    )


@pytest.mark.asyncio()
async def test_generate_endpoint(mock_generate_image_b64, red_b64):
    test_input = {
        "input": {
            "prompts": [{"text": "A dog"}],
            "sampler": {"id": "ddim"},
            "height": 512,
            "width": 512,
        },
    }

    response = client.post("/api/stablestudio/generate", json=test_input)

    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    for image in data["images"]:
        assert image["blob"] == red_b64.decode("utf-8")


@pytest.mark.asyncio()
async def test_list_samplers():
    response = client.get("/api/stablestudio/samplers")
    assert response.status_code == 200
    assert response.json() == [
        {"id": "ddim", "name": "ddim"},
        {"id": "dpmpp", "name": "dpmpp"},
    ]


@pytest.mark.asyncio()
async def test_list_models():
    response = client.get("/api/stablestudio/models")
    assert response.status_code == 200

    expected_model_ids = {
        "sd15",
        "openjourney-v1",
        "openjourney-v2",
        "openjourney-v4",
        "modern-disney",
        "redshift-diffusion",
        "sdxl",
        "opendalle11",
    }
    model_ids = {m["id"] for m in response.json()}
    assert model_ids == expected_model_ids

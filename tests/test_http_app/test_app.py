from unittest import mock

import pytest
from fastapi.testclient import TestClient

from imaginairy.http_app.app import app

client = TestClient(app)


@pytest.fixture()
def mock_generate_image(monkeypatch):
    fake_generate = mock.MagicMock(return_value=iter("a fake image"))
    monkeypatch.setattr("imaginairy.http_app.app.generate_image", fake_generate)


@pytest.mark.asyncio()
def test_imagine_endpoint(mock_generate_image):
    test_input = {"prompt": "test prompt"}

    response = client.post("/api/imagine", json=test_input)

    assert response.status_code == 200
    assert response.content == b"a fake image"


@pytest.mark.asyncio()
async def test_get_imagine_endpoint(mock_generate_image):
    test_input = {"text": "a dog"}

    response = client.get("/api/imagine", params=test_input)

    assert response.status_code == 200
    assert response.content == b"a fake image"


@pytest.mark.asyncio()
async def test_get_imagine_endpoint_mp(mock_generate_image):
    test_input = {"text": "a dog"}

    response = client.get("/api/imagine", params=test_input)

    assert response.status_code == 200
    assert response.content == b"a fake image"


def test_edit_redir():
    response = client.get("/edit")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.content[:15] == b"<!DOCTYPE html>"


def test_generate_redir():
    response = client.get("/generate")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.content[:15] == b"<!DOCTYPE html>"

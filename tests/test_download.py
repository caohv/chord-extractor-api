from fastapi.testclient import TestClient

from app.main import app


def test_download_rejects_invalid_url() -> None:
    client = TestClient(app)
    resp = client.post("/download", json={"url": "not-a-url"})
    assert resp.status_code == 422


def test_download_route_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/download" in paths

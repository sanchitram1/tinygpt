from conftest import FakeStoryGenerator


def test_rate_limit_returns_429(make_client):
    client = make_client(generator=FakeStoryGenerator(), rate_limit_per_minute=2)
    for _ in range(2):
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"


def test_rate_limit_zero_disables(make_client):
    client = make_client(generator=FakeStoryGenerator(), rate_limit_per_minute=0)
    for _ in range(5):
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200


def test_oversized_body_rejected_before_parsing(make_client):
    client = make_client(generator=FakeStoryGenerator(), max_body_bytes=50)
    response = client.post("/api/chat", json={"message": "x" * 100})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_body_limit_does_not_affect_health_or_static(make_client):
    client = make_client(generator=FakeStoryGenerator(), max_body_bytes=10)
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200

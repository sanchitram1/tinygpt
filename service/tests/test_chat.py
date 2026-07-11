import threading

from conftest import CaptureTraceSink, FailingGenerator, FakeStoryGenerator


def test_chat_happy_path(make_client):
    generator = FakeStoryGenerator()
    sink = CaptureTraceSink()
    client = make_client(generator=generator, sink=sink)

    response = client.post("/api/chat", json={"message": "a brave little fox"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Once upon a time, a brave little fox"
    assert body["model_version"] == "xlarge-plus"
    assert body["run_id"] == "test-run-001"
    assert body["stop_reason"] == "eos"
    assert body["session_id"]  # server-generated UUID when absent
    assert body["request_id"]
    assert body["synthetic"] is False
    # defaults applied
    assert body["decoding"] == {"temperature": 0.7, "top_k": 30, "max_new_tokens": 200}
    assert len(sink.records) == 1
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_chat_round_trips_session_id_and_decoding(make_client):
    generator = FakeStoryGenerator()
    client = make_client(generator=generator)
    response = client.post(
        "/api/chat",
        json={
            "message": "hi",
            "session_id": "abc-123",
            "temperature": 0.5,
            "top_k": 10,
            "max_new_tokens": 64,
        },
    )
    body = response.json()
    assert body["session_id"] == "abc-123"
    assert body["decoding"] == {"temperature": 0.5, "top_k": 10, "max_new_tokens": 64}
    assert generator.calls[0]["temperature"] == 0.5


def test_max_new_tokens_clamped_to_server_limit(make_client):
    generator = FakeStoryGenerator()
    client = make_client(generator=generator, max_new_tokens_limit=100)
    response = client.post("/api/chat", json={"message": "hi", "max_new_tokens": 9999})
    assert response.status_code == 200
    assert response.json()["decoding"]["max_new_tokens"] == 100
    assert generator.calls[0]["max_new_tokens"] == 100


def test_invalid_bodies_get_stable_error(client):
    for payload in (
        {},  # missing message
        {"message": ""},  # empty
        {"message": "hi", "temperature": -1},
        {"message": "hi", "session_id": "bad session id!"},
        {"message": "hi", "unexpected": True},  # extra fields forbidden
    ):
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 422, payload
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["message"]


def test_message_over_configured_limit_rejected(make_client):
    client = make_client(generator=FakeStoryGenerator(), max_message_chars=10)
    response = client.post("/api/chat", json={"message": "x" * 11})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_chat_503_when_model_unavailable(make_client):
    sink = CaptureTraceSink()
    client = make_client(generator=None, sink=sink)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    # failed generations still produce a trace
    assert sink.records[0]["error.type"] == "model_unavailable"
    assert sink.records[0]["http.status_code"] == 503


def test_chat_503_when_generation_slot_busy(make_client):
    sink = CaptureTraceSink()
    client = make_client(
        generator=FakeStoryGenerator(), sink=sink, generation_wait_seconds=0.05
    )
    semaphore: threading.Semaphore = client.app_state.generation_semaphore
    assert semaphore.acquire(timeout=1)
    try:
        response = client.post("/api/chat", json={"message": "hi"})
    finally:
        semaphore.release()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_busy"
    assert sink.records[0]["error.type"] == "model_busy"


def test_chat_500_when_generation_raises(make_client):
    sink = CaptureTraceSink()
    client = make_client(generator=FailingGenerator(), sink=sink)
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "generation_failed"
    assert "cuda exploded" not in response.text  # internals not leaked
    assert sink.records[0]["error.type"] == "RuntimeError"
    # semaphore released after failure: a follow-up request is not "busy"
    client2_response = client.post("/api/chat", json={"message": "hi"})
    assert client2_response.json()["error"]["code"] == "generation_failed"


def test_synthetic_flag_propagates(make_client):
    sink = CaptureTraceSink()
    client = make_client(generator=FakeStoryGenerator(), sink=sink)
    response = client.post("/api/chat", json={"message": "hi", "synthetic": True})
    assert response.json()["synthetic"] is True
    assert sink.records[0]["synthetic"] is True

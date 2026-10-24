from __future__ import annotations

import json

import pytest

from src import openai_compatible as adapter


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(
        self,
        chunks: list[dict] | None = None,
        *,
        include_done: bool = True,
        status_code: int = 200,
        body: dict | None = None,
    ):
        self.status_code = status_code
        self.body = body
        self.text = json.dumps(body) if body else ""
        self.lines = [f"data: {json.dumps(chunk)}".encode() for chunk in chunks or []]
        if include_done:
            self.lines.append(b"data: [DONE]")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_lines(self, chunk_size=None):
        return iter(self.lines)

    def json(self):
        if self.body is None:
            raise ValueError("no JSON body")
        return self.body

    def close(self):
        pass


class FakeSession:
    def __init__(self, response: FakeResponse | list[FakeResponse]):
        self.headers = {}
        self.responses = response if isinstance(response, list) else [response]
        self.request = None
        self.closed = False
        self.post_count = 0

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        response = self.responses[self.post_count]
        self.post_count += 1
        return response

    def close(self):
        self.closed = True


def test_normalize_base_url_preserves_path_prefix_and_rejects_credentials():
    assert adapter.normalize_base_url("https://host.example/proxy") == (
        "https://host.example/proxy/v1"
    )
    assert adapter.normalize_base_url("https://host.example/proxy/v1/") == (
        "https://host.example/proxy/v1"
    )
    with pytest.raises(ValueError, match="credentials"):
        adapter.normalize_base_url("https://name:secret@host.example/v1")
    with pytest.raises(ValueError, match="query"):
        adapter.normalize_base_url("https://host.example/v1?token=secret")


def test_run_streams_answer_and_records_provider_usage(monkeypatch):
    response = FakeResponse([
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 11, "completion_tokens": 3}},
    ])
    session = FakeSession(response)
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/api/v1")
    monkeypatch.setattr(adapter, "create_http_session", lambda **_kwargs: session)
    monkeypatch.setattr(adapter, "_stagger_delay", lambda *_args: None)

    results = adapter.run_openai_compatible(
        model="vendor/model",
        prompt_category="general",
        mode="warm",
        num_runs=1,
        custom_system_prompt="Use the task instruction.",
        custom_tasks=[{"id": "task-1", "prompt": "question"}],
        store_responses=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result["response"] == "answer"
    assert result["error"] is None
    assert result["ttft_s"] >= 0
    assert result["e2e_s"] >= result["ttft_s"]
    assert result["prompt_tokens"] == 11
    assert result["output_tokens"] == 3
    assert result["output_token_source"] == "provider_usage"
    assert result["model_memory_mb"] is None
    assert session.headers["Authorization"] == "Bearer provider-secret"
    assert session.request[0] == "https://provider.example/api/v1/chat/completions"
    assert session.request[1]["json"]["max_completion_tokens"] == 512
    assert "provider-secret" not in json.dumps(result)
    assert session.closed


def test_run_rejects_stream_without_terminal_marker(monkeypatch):
    session = FakeSession(FakeResponse([
        {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
    ], include_done=False))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setattr(adapter, "create_http_session", lambda **_kwargs: session)
    monkeypatch.setattr(adapter, "_stagger_delay", lambda *_args: None)

    result = adapter.run_openai_compatible(
        model="vendor/model",
        prompt_category="general",
        mode="warm",
        num_runs=1,
        custom_system_prompt="system",
        custom_tasks=[{"id": "task-1", "prompt": "question"}],
    )[0]

    assert result["response"] == ""
    assert "before [DONE]" in result["error"]
    assert result["output_tokens"] is None


def test_run_falls_back_to_legacy_token_limit_for_compatible_servers(monkeypatch):
    session = FakeSession([
        FakeResponse(status_code=400, body={
            "error": {"message": "Unsupported parameter: max_completion_tokens"}
        }),
        FakeResponse([
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
            {"usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        ]),
    ])
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setattr(adapter, "create_http_session", lambda **_kwargs: session)
    monkeypatch.setattr(adapter, "_stagger_delay", lambda *_args: None)

    result = adapter.run_openai_compatible(
        model="vendor/model",
        prompt_category="general",
        mode="warm",
        num_runs=1,
        custom_system_prompt="system",
        custom_tasks=[{"id": "task-1", "prompt": "question"}],
    )[0]

    assert result["error"] is None
    assert session.post_count == 2
    assert session.request[1]["json"]["max_tokens"] == 512
    assert "max_completion_tokens" not in session.request[1]["json"]


def test_detect_requires_key_without_network_request(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert adapter.detect_openai_compatible() is False
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    assert adapter.detect_openai_compatible() is True

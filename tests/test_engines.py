import json
from unittest.mock import patch

import requests
import pytest

from src.engines import (
    _reasoning_delta,
    _stream_decode_tps,
    _stream_text,
    detect_vllm,
    normalize_vllm_api_base_url,
    post_with_retry,
    stream_request_timeout,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}", response=self
            )
        return None

    def json(self):
        return self.payload


class FakeVllmSession:
    def __init__(self, models_status: int = 200, probe_status: int = 200):
        self.models_status = models_status
        self.probe_status = probe_status
        self.calls: list[tuple[str, str]] = []

    def get(self, url, **_kwargs):
        self.calls.append(("GET", url))
        return FakeResponse(
            self.models_status,
            {"data": [{"id": "served-model", "max_model_len": 8192}]},
        )

    def post(self, url, **_kwargs):
        self.calls.append(("POST", url))
        return FakeResponse(self.probe_status)


@pytest.fixture
def vllm_preflight(monkeypatch):
    session = FakeVllmSession()
    monkeypatch.setattr("src.engines.create_http_session", lambda **_kwargs: session)
    return session


def test_stream_decode_tps_excludes_first_token() -> None:
    assert _stream_decode_tps(7, 2.0) == 3.0
    assert _stream_decode_tps(1, 2.0) is None
    assert _stream_decode_tps(None, 2.0) is None


def test_stream_request_timeout_preserves_safety_timeout_when_deadline_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.engines.time.perf_counter", lambda: 500.0)

    assert stream_request_timeout(0, 0.0) == (20.0, 120.0)
    assert stream_request_timeout(30, 490.0) == (20.0, 20.0)
    assert stream_request_timeout(600, 0.0) == (20.0, 100.0)
    assert stream_request_timeout(700, 0.0) == (20.0, 120.0)


def test_openai_compatible_stream_fields_support_current_legacy_and_blocks() -> None:
    assert _stream_text([{"type": "text", "text": "bon"}, {"content": "jour"}]) == "bonjour"
    assert _reasoning_delta({"reasoning": "current"}) == "current"
    assert _reasoning_delta({"reasoning_content": "legacy"}) == "legacy"
    assert _reasoning_delta({"thinking": "compatible"}) == "compatible"


def test_detect_vllm_requires_vllm_version_and_models_endpoints() -> None:
    with patch("src.engines.requests.get", side_effect=[FakeResponse(200), FakeResponse(200)]) as get:
        assert detect_vllm() is True

    assert get.call_args_list[0].args[0].endswith("/version")
    assert get.call_args_list[1].args[0].endswith("/v1/models")


def test_detect_vllm_rejects_openai_compatible_non_vllm_server() -> None:
    with patch("src.engines.requests.get", side_effect=[FakeResponse(404), FakeResponse(200)]):
        assert detect_vllm() is False


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
        ("https://gateway.example/llm/v1", "https://gateway.example/llm/v1"),
        ("https://gateway.example/llm", "https://gateway.example/llm/v1"),
    ],
)
def test_normalize_vllm_api_base_url(configured: str, expected: str) -> None:
    assert normalize_vllm_api_base_url(configured) == expected


@pytest.mark.parametrize("configured", ["", "localhost:8000", "ftp://localhost/v1"])
def test_normalize_vllm_api_base_url_rejects_invalid_values(configured: str) -> None:
    with pytest.raises(ValueError, match="absolute http"):
        normalize_vllm_api_base_url(configured)


def test_detect_vllm_normalizes_a_server_root(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/")
    with patch(
        "src.engines.requests.get",
        side_effect=[FakeResponse(200), FakeResponse(200)],
    ) as get:
        assert detect_vllm() is True

    assert get.call_args_list[0].args[0] == "http://127.0.0.1:8000/version"
    assert get.call_args_list[1].args[0] == "http://127.0.0.1:8000/v1/models"


def test_run_vllm_uses_usage_completion_tokens_not_stream_chunk_count(
    monkeypatch, vllm_preflight
) -> None:
    from src.engines import run_vllm

    chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}).encode() + b"\n\n",
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "a"}}]}).encode() + b"\n\n",
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "b"}}]}).encode() + b"\n\n",
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(chunks)

    task_urls: list[str] = []

    def fake_post(_session, url, **_kwargs):
        task_urls.append(url)
        return Response()

    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    with patch("src.engines.post_with_retry", side_effect=fake_post):
        result = run_vllm(
            "served-model",
            "reasoning",
            "warm",
            1,
            custom_tasks=[{"id": "reasoning-1", "prompt": "What is 2 + 2?"}],
        )

    assert result[0]["output_tokens"] == 7
    assert result[0]["prompt_tokens"] == 5
    assert result[0]["output_token_source"] == "vllm_usage"
    assert task_urls == ["http://127.0.0.1:8000/v1/chat/completions"]


def test_run_vllm_does_not_retry_completion_budget_exhaustion(
    vllm_preflight,
) -> None:
    from src.engines import COMPLETION_BUDGET_EXHAUSTED, run_vllm

    reasoning_only_chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"reasoning_content": "thinking"}}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}).encode(),
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 256}}).encode(),
        b"data: [DONE]",
    ]
    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __init__(self, chunks):
            self.chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(self.chunks)

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response(reasoning_only_chunks)

    with patch("src.engines.post_with_retry", side_effect=fake_post):
        result = run_vllm(
            "served-model",
            "fr_fact_verification",
            "warm",
            1,
            store_responses=True,
            max_output_tokens=256,
            custom_tasks=[{"id": "fact-1", "prompt": "Classify this claim."}],
        )

    assert len(payloads) == 1
    assert payloads[0]["max_tokens"] == 256
    assert all("chat_template_kwargs" not in payload for payload in payloads)
    assert COMPLETION_BUDGET_EXHAUSTED in result[0]["error"]
    assert result[0]["response"] == ""


def test_run_vllm_stops_large_suite_after_systematic_budget_exhaustion(
    vllm_preflight,
) -> None:
    from src.engines import COMPLETION_BUDGET_EXHAUSTED, run_vllm

    chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"reasoning_content": "thinking"}}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}).encode(),
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 256}}).encode(),
        b"data: [DONE]",
    ]
    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(chunks)

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    with patch("src.engines.post_with_retry", side_effect=fake_post):
        result = run_vllm(
            "served-model",
            "fr_fact_verification",
            "warm",
            1,
            concurrency=2,
            max_output_tokens=256,
            custom_tasks=[
                {"id": f"fact-{index}", "prompt": "Classify this claim."}
                for index in range(20)
            ],
        )

    assert len(result) == 2
    assert len(payloads) == 2
    assert all(COMPLETION_BUDGET_EXHAUSTED in item["error"] for item in result)


def test_run_vllm_enforces_generation_timeout_without_retry(
    monkeypatch, vllm_preflight
) -> None:
    from src.engines import GENERATION_TIMEOUT_EXCEEDED, run_vllm

    chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "still generating"}}]}).encode(),
        b"data: [DONE]",
    ]
    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(chunks)

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    tick = iter(float(value) for value in range(20))
    monkeypatch.setattr("src.engines.time.perf_counter", lambda: next(tick))
    monkeypatch.setattr("src.engines.post_with_retry", fake_post)

    result = run_vllm(
        "served-model",
        "fr_fact_verification",
        "warm",
        1,
        concurrency=1,
        generation_timeout_seconds=1,
        custom_tasks=[{"id": "fact-0", "prompt": "Classify this claim."}],
    )

    assert len(payloads) == 1
    assert len(result) == 1
    assert GENERATION_TIMEOUT_EXCEEDED in result[0]["error"]


def test_run_vllm_silent_stream_uses_remaining_deadline_as_read_timeout(
    monkeypatch, vllm_preflight
) -> None:
    from src.engines import GENERATION_TIMEOUT_EXCEEDED, run_vllm

    observed_timeouts: list[object] = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            raise requests.exceptions.ReadTimeout("read timed out")

    def fake_post(_session, _url, *, timeout, **_kwargs):
        observed_timeouts.append(timeout)
        return Response()

    monkeypatch.setattr("src.engines.post_with_retry", fake_post)
    result = run_vllm(
        "served-model",
        "fr_fact_verification",
        "warm",
        1,
        generation_timeout_seconds=7,
        custom_tasks=[{"id": "fact-0", "prompt": "Classify this claim."}],
    )

    assert len(observed_timeouts) == 1
    assert isinstance(observed_timeouts[0], tuple)
    assert 0 < observed_timeouts[0][1] <= 7
    assert GENERATION_TIMEOUT_EXCEEDED in result[0]["error"]


def test_run_vllm_stops_category_when_server_dies_mid_stream(monkeypatch) -> None:
    from src.engines import ENGINE_UNAVAILABLE, run_vllm

    class Session(FakeVllmSession):
        def get(self, url, **kwargs):
            if url.endswith("/health"):
                raise requests.exceptions.ConnectionError("server exited")
            return super().get(url, **kwargs)

        def close(self):
            return None

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            raise requests.exceptions.ConnectionError("stream closed")

    payloads: list[dict] = []

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    monkeypatch.setattr("src.engines.create_http_session", lambda **_kwargs: Session())
    monkeypatch.setattr("src.engines.post_with_retry", fake_post)
    result = run_vllm(
        "served-model",
        "fr_fact_verification",
        "warm",
        1,
        concurrency=2,
        custom_tasks=[
            {"id": f"fact-{index}", "prompt": "Classify this claim."}
            for index in range(6)
        ],
    )

    assert len(payloads) == 2
    assert len(result) == 2
    assert all(ENGINE_UNAVAILABLE in item["error"] for item in result)


def test_run_vllm_continues_after_isolated_generation_timeout(
    monkeypatch, vllm_preflight
) -> None:
    from src.engines import GENERATION_TIMEOUT_EXCEEDED, run_vllm

    success_chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "still generating"}}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode(),
        b"data: [DONE]",
    ]
    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __init__(self, times_out: bool):
            self.times_out = times_out

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            if self.times_out:
                raise RuntimeError(GENERATION_TIMEOUT_EXCEEDED)
            return iter(success_chunks)

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response(times_out=len(payloads) == 1)

    monkeypatch.setattr("src.engines.post_with_retry", fake_post)

    result = run_vllm(
        "served-model",
        "fr_fact_verification",
        "warm",
        1,
        concurrency=2,
        generation_timeout_seconds=1,
        custom_tasks=[
            {"id": f"fact-{index}", "prompt": "Classify this claim."}
            for index in range(4)
        ],
    )

    assert len(payloads) == 4
    assert len(result) == 4
    assert sum(GENERATION_TIMEOUT_EXCEEDED in str(item.get("error") or "") for item in result) == 1
    assert sum(item.get("error") is None for item in result) == 3


def test_run_vllm_stops_after_full_batch_generation_timeout(
    monkeypatch, vllm_preflight
) -> None:
    from src.engines import GENERATION_TIMEOUT_EXCEEDED, run_vllm

    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            raise RuntimeError(GENERATION_TIMEOUT_EXCEEDED)

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    monkeypatch.setattr("src.engines.post_with_retry", fake_post)

    result = run_vllm(
        "served-model",
        "fr_fact_verification",
        "warm",
        1,
        concurrency=2,
        generation_timeout_seconds=1,
        custom_tasks=[
            {"id": f"fact-{index}", "prompt": "Classify this claim."}
            for index in range(4)
        ],
    )

    assert len(payloads) == 2
    assert len(result) == 2
    assert all(GENERATION_TIMEOUT_EXCEEDED in item["error"] for item in result)


def test_run_vllm_generation_timeout_can_be_disabled(
    monkeypatch, vllm_preflight
) -> None:
    from itertools import count

    from src.engines import run_vllm

    chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode(),
        b"data: [DONE]",
    ]

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(chunks)

    elapsed = count(0.0, 10_000.0)
    monkeypatch.setattr("src.engines.time.perf_counter", lambda: next(elapsed))
    monkeypatch.setattr(
        "src.engines.post_with_retry", lambda *_args, **_kwargs: Response()
    )

    result = run_vllm(
        "served-model",
        "fr_fact_verification",
        "warm",
        1,
        store_responses=True,
        generation_timeout_seconds=0,
        custom_tasks=[{"id": "fact-1", "prompt": "Classify this claim."}],
    )

    assert len(result) == 1
    assert result[0]["error"] is None
    assert result[0]["response"] == "ok"


def test_run_vllm_rejects_empty_assistant_response(vllm_preflight) -> None:
    from src.engines import run_vllm

    chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}}).encode(),
        b"data: [DONE]",
    ]

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(chunks)

    with patch("src.engines.post_with_retry", return_value=Response()):
        result = run_vllm(
            "served-model",
            "fr_fact_verification",
            "warm",
            1,
            store_responses=True,
            custom_tasks=[{"id": "fact-1", "prompt": "Classify this claim."}],
        )

    assert "empty assistant response" in result[0]["error"]
    assert result[0]["response"] == ""


def test_run_vllm_preserves_native_reasoning_for_each_inflight_request(
    vllm_preflight,
) -> None:
    from src.engines import run_vllm

    reasoning_chunks = [
        b'data: ' + json.dumps({"choices": [{"delta": {"reasoning_content": "thinking"}}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "answer"}}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode(),
        b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode(),
        b"data: [DONE]",
    ]
    payloads: list[dict] = []

    class Response:
        status_code = 200

        def __init__(self, chunks):
            self.chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(self.chunks)

    def fake_post(_session, _url, *, json, **_kwargs):
        payloads.append(json)
        return Response(reasoning_chunks)

    with patch("src.engines.post_with_retry", side_effect=fake_post):
        result = run_vllm(
            "served-model",
            "dynamic-category",
            "warm",
            1,
            concurrency=2,
            store_responses=True,
            custom_tasks=[
                {"id": "task-1", "prompt": "one"},
                {"id": "task-2", "prompt": "two"},
            ],
        )

    assert len(payloads) == 2
    assert all("chat_template_kwargs" not in payload for payload in payloads)
    assert all(item["error"] is None and item["response"] == "answer" for item in result)
    assert all(item["generation_ttft_s"] <= item["ttft_s"] <= item["e2e_s"] for item in result)
    assert all(item["reasoning_s"] is not None for item in result)


def test_run_vllm_rejects_false_cold_mode_provenance() -> None:
    from src.engines import run_vllm

    with pytest.raises(ValueError, match="only mode='warm'"):
        run_vllm("served-model", "quantitative", "cold", 1)


def test_run_vllm_rejects_external_retrieval_before_requests(monkeypatch) -> None:
    from src import engines

    def unexpected_session(**kwargs):
        pytest.fail("retrieval task must fail before creating a network session")

    monkeypatch.setattr(engines, "create_http_session", unexpected_session)
    with pytest.raises(ValueError, match="External retrieval is not supported"):
        engines.run_vllm(
            model="served-model", prompt_category="general", mode="warm", num_runs=1,
            custom_tasks=[{"id": "retrieval", "prompt": "Question", "use_rag": True}],
        )


def test_run_vllm_fails_before_task_pool_when_models_route_is_missing(
    monkeypatch,
) -> None:
    from src.engines import run_vllm

    session = FakeVllmSession(models_status=404)
    monkeypatch.setattr("src.engines.create_http_session", lambda **_kwargs: session)
    with patch("src.engines.post_with_retry") as task_post:
        with pytest.raises(RuntimeError, match="vLLM preflight failed.*v1/models"):
            run_vllm(
                "served-model",
                "dynamic-category",
                "warm",
                1,
                custom_tasks=[{"id": "task-1", "prompt": "question"}],
            )

    task_post.assert_not_called()


def test_run_vllm_fails_before_task_pool_when_chat_route_is_missing(
    monkeypatch,
) -> None:
    from src.engines import run_vllm

    session = FakeVllmSession(probe_status=404)
    monkeypatch.setattr("src.engines.create_http_session", lambda **_kwargs: session)
    with patch("src.engines.post_with_retry") as task_post:
        with pytest.raises(RuntimeError, match="chat preflight failed.*v1/chat/completions"):
            run_vllm(
                "served-model",
                "dynamic-category",
                "warm",
                1,
                custom_tasks=[{"id": "task-1", "prompt": "question"}],
            )

    task_post.assert_not_called()


# ---------------------------------------------------------------------------
# post_with_retry — bounded timeout + bounded retries
# ---------------------------------------------------------------------------

class RetrySession:
    """Fake session: fails with the given exception/status `fail_times` times
    before returning the success response."""

    def __init__(self, fail_times: int, failure, success_status: int = 200):
        self.fail_times = fail_times
        self.failure = failure
        self.success_status = success_status
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            if isinstance(self.failure, Exception):
                raise self.failure
            return FakeResponse(self.failure)
        return FakeResponse(self.success_status)


class TestPostWithRetry:
    def test_success_first_try_no_retry(self) -> None:
        session = RetrySession(0, None)
        resp = post_with_retry(session, "http://x/chat", retries=2, backoff=0)
        assert resp.status_code == 200
        assert session.calls == 1

    def test_retries_on_connection_error(self) -> None:
        session = RetrySession(2, requests.exceptions.ConnectionError("refused"))
        resp = post_with_retry(session, "http://x/chat", retries=2, backoff=0)
        assert resp.status_code == 200
        assert session.calls == 3

    def test_retries_on_timeout(self) -> None:
        session = RetrySession(1, requests.exceptions.Timeout("hung"))
        resp = post_with_retry(session, "http://x/chat", retries=2, backoff=0)
        assert resp.status_code == 200
        assert session.calls == 2

    def test_retries_on_5xx(self) -> None:
        session = RetrySession(2, 503)
        resp = post_with_retry(session, "http://x/chat", retries=2, backoff=0)
        assert resp.status_code == 200
        assert session.calls == 3

    def test_does_not_retry_4xx(self) -> None:
        session = RetrySession(1, 400)
        resp = post_with_retry(session, "http://x/chat", retries=2, backoff=0)
        assert resp.status_code == 400
        assert session.calls == 1

    def test_raises_after_exhausting_retries(self) -> None:
        session = RetrySession(5, requests.exceptions.ConnectionError("down"))
        try:
            post_with_retry(session, "http://x/chat", retries=2, backoff=0)
            assert False, "should have raised"
        except requests.exceptions.ConnectionError:
            pass
        assert session.calls == 3

    def test_raises_http_error_when_5xx_exhausted(self) -> None:
        session = RetrySession(5, 500)
        try:
            post_with_retry(session, "http://x/chat", retries=2, backoff=0)
            assert False, "should have raised"
        except requests.exceptions.HTTPError:
            pass
        assert session.calls == 3

    def test_uses_bounded_timeout(self) -> None:
        session = RetrySession(0, None)
        post_with_retry(session, "http://x/chat", retries=0)
        assert session.calls == 1

from pathlib import Path

import pytest
import requests

from src import benchmark


def test_summarise_framework_labeling_matches_engine() -> None:
    runs = [{
        "ttft_s": 0.1,
        "e2e_s": 1.0,
        "prefill_tps": 100.0,
        "decode_tps": 50.0,
        "model_memory_mb": 1024,
        "prompt_tokens": 10,
        "output_tokens": 50,
    }]

    summary = benchmark.summarise("model", "faq", "warm", runs, engine="ollama")

    assert summary["framework"] == "ollama"


def test_summarise_reports_generation_timeout_count_and_rate() -> None:
    runs = [
        {
            "ttft_s": 0.1,
            "e2e_s": 1.0,
            "prefill_tps": 100.0,
            "decode_tps": 50.0,
            "model_memory_mb": 1024,
            "prompt_tokens": 10,
            "output_tokens": 50,
        },
        {
            "error": "generation timeout exceeded before a complete assistant answer "
            "(120s limit)"
        },
    ]

    summary = benchmark.summarise("model", "faq", "warm", runs, engine="vllm")

    assert summary["error_count"] == 1
    assert summary["generation_timeout_count"] == 1
    assert summary["generation_timeout_rate"] == 0.5


def test_unload_model_surfaces_ollama_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("unload failed")

    def post(path: str, **_):
        called.append(path)
        return Response()

    monkeypatch.setattr(benchmark.CLIENT, "post", post)

    with pytest.raises(requests.HTTPError):
        benchmark.unload_model("any-model")
    assert called == ["/api/chat"]


def test_warm_model_uses_chat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

    def post(path: str, **kwargs):
        request["path"] = path
        request.update(kwargs)
        return Response()

    monkeypatch.setattr(benchmark.CLIENT, "post", post)
    benchmark.warm_model("any-model")

    assert request["path"] == "/api/chat"
    assert request["json"] == {
        "model": "any-model",
        "messages": [{"role": "user", "content": ""}],
        "stream": False,
        "keep_alive": "5m",
        "options": {"num_predict": 1},
    }


def test_measurement_omits_full_response_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    response = {"message": {"content": "safe preview"}, "done": True, "eval_count": 1, "eval_duration": 1}

    class StreamResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            yield b'{"message":{"content":"safe preview"},"done":true,"eval_count":1,"eval_duration":1}'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(benchmark.CLIENT, "post", lambda *args, **kwargs: StreamResponse())
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)

    result = benchmark._measure_once("model", "system", "user", False, 1, 0, "task")
    assert result["response_preview"] == "safe preview"


def test_measurement_applies_ollama_output_token_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []

    class StreamResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            yield b'{"message":{"content":"ok"},"done":true,"eval_count":1,"eval_duration":1}'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def post(*_args, **kwargs):
        payloads.append(kwargs.get("json", {}))
        return StreamResponse()

    monkeypatch.setattr(benchmark.CLIENT, "post", post)
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)

    benchmark._measure_once(
        "model", "system", "user", False, 1, 0, "task", max_output_tokens=128
    )

    assert payloads[-1]["options"]["num_predict"] == 128


def test_ollama_keeps_native_reasoning_and_charges_it_to_answer_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []
    stream = [
        b'{"message":{"thinking":"reasoning"},"done":false}',
        b'{"message":{"content":"final"},"done":true,"done_reason":"stop","eval_count":6,"eval_duration":6}',
    ]

    class StreamResponse:
        status_code = 200

        def __init__(self, chunks):
            self.chunks = chunks

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(self.chunks)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def post(*_args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        return StreamResponse(stream)

    monkeypatch.setattr(benchmark.CLIENT, "post", post)
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)

    result = benchmark._measure_once(
        "model", "system", "user", False, 1, 0, "task", store_response=True
    )

    assert len(payloads) == 1
    assert "think" not in payloads[0]
    assert result["response"] == "final"
    assert result["generation_ttft_s"] <= result["ttft_s"] <= result["e2e_s"]
    assert result["reasoning_s"] is not None
    assert result["attempt_count"] == 1


def test_ollama_does_not_retry_completion_budget_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_budgets: list[int] = []
    stream = [
        b'{"message":{"thinking":"reasoning"},"done":false}',
        b'{"message":{},"done":true,"done_reason":"length","prompt_eval_count":5,"prompt_eval_duration":5,"eval_count":256,"eval_duration":256}',
    ]

    class StreamResponse:
        status_code = 200

        def __init__(self, chunks):
            self.chunks = chunks

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            return iter(self.chunks)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def post(*_args, **kwargs):
        requested_budgets.append(int(kwargs["json"]["options"]["num_predict"]))
        return StreamResponse(stream)

    monkeypatch.setattr(benchmark.CLIENT, "post", post)
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)

    with pytest.raises(RuntimeError, match=benchmark.COMPLETION_BUDGET_EXHAUSTED):
        benchmark._measure_once(
            "model",
            "system",
            "user",
            False,
            1,
            0,
            "task",
            store_response=True,
            min_ctx=4096,
            max_output_tokens=256,
        )

    assert requested_budgets == [256]


def test_ollama_generation_timeout_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    post_calls = 0

    class StreamResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            yield b'{"message":{"thinking":"still generating"},"done":false}'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def post(*_args, **_kwargs):
        nonlocal post_calls
        post_calls += 1
        return StreamResponse()

    ticks = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(benchmark.CLIENT, "post", post)
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)

    with pytest.raises(RuntimeError, match=benchmark.GENERATION_TIMEOUT_EXCEEDED):
        benchmark._measure_once(
            "model",
            "system",
            "user",
            False,
            1,
            0,
            "task",
            generation_timeout_seconds=1,
        )

    assert post_calls == 1


def test_ollama_silent_stream_is_bounded_by_generation_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[object] = []

    class StreamResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            raise requests.exceptions.ReadTimeout("read timed out")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def post(*_args, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return StreamResponse()

    monkeypatch.setattr(benchmark.CLIENT, "post", post)
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)

    with pytest.raises(RuntimeError, match=benchmark.GENERATION_TIMEOUT_EXCEEDED):
        benchmark._measure_once(
            "model",
            "system",
            "user",
            False,
            1,
            0,
            "task",
            generation_timeout_seconds=9,
        )

    assert len(observed_timeouts) == 1
    assert isinstance(observed_timeouts[0], tuple)
    assert 0 < observed_timeouts[0][1] <= 9


def test_ollama_generation_timeout_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class StreamResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, *args, **kwargs):
            yield b'{"message":{"content":"ok"},"done":true,"done_reason":"stop","eval_count":1,"eval_duration":1}'

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    elapsed = iter([0.0, 10_000.0, 10_001.0, 10_002.0, 10_003.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(elapsed))
    monkeypatch.setattr(benchmark.CLIENT, "post", lambda *_args, **_kwargs: StreamResponse())
    monkeypatch.setattr(benchmark, "loaded_model_memory_mb", lambda _: None)

    result = benchmark._measure_once(
        "model",
        "system",
        "user",
        False,
        1,
        0,
        "task",
        generation_timeout_seconds=0,
    )

    assert result["error"] is None
    assert result["response_preview"] == "ok"


def test_ollama_quality_suite_continues_after_isolated_generation_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def measure(*_args, **kwargs):
        task_id = str(kwargs.get("task_id") or _args[6])
        calls.append(task_id)
        if task_id == "task-0":
            raise RuntimeError(benchmark.GENERATION_TIMEOUT_EXCEEDED)
        return {
            "run_index": kwargs.get("run_index", 1),
            "req_index": kwargs.get("req_index", 0),
            "cold": False,
            "ttft_s": 0.1,
            "e2e_s": 0.2,
            "prefill_tps": 10.0,
            "decode_tps": 10.0,
            "prompt_tokens": 5,
            "output_tokens": 2,
            "model_memory_mb": None,
            "response_preview": "ok",
            "response": "ok",
            "task_id": task_id,
            "error": None,
        }

    monkeypatch.setattr(benchmark, "_measure_once", measure)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)
    tasks = {
        f"task-{index}": {"id": f"task-{index}", "prompt": "prompt"}
        for index in range(20)
    }

    results = benchmark.run_quality_suite(
        model="model",
        mode="warm",
        concurrency=1,
        tasks=tasks,
        repetitions=1,
        generation_timeout_seconds=1,
    )

    assert len(calls) == 20
    assert len(results) == 20
    assert sum(benchmark.GENERATION_TIMEOUT_EXCEEDED in str(result.get("error") or "") for result in results) == 1
    assert sum(result.get("error") is None for result in results) == 19


def test_ollama_quality_suite_stops_after_full_batch_generation_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_measure(*_args, **kwargs):
        calls.append(str(kwargs.get("task_id") or _args[6]))
        raise RuntimeError(benchmark.GENERATION_TIMEOUT_EXCEEDED)

    monkeypatch.setattr(benchmark, "_measure_once", fail_measure)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)
    tasks = {
        f"task-{index}": {"id": f"task-{index}", "prompt": "prompt"}
        for index in range(20)
    }

    results = benchmark.run_quality_suite(
        model="model",
        mode="warm",
        concurrency=2,
        tasks=tasks,
        repetitions=2,
        generation_timeout_seconds=1,
    )

    assert len(calls) == 2
    assert len(results) == 2
    assert all(benchmark.GENERATION_TIMEOUT_EXCEEDED in result["error"] for result in results)


def test_ollama_quality_suite_stops_after_systematic_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_measure(*_args, **kwargs):
        calls.append(str(kwargs.get("task_id") or _args[6]))
        raise RuntimeError(benchmark.COMPLETION_BUDGET_EXHAUSTED)

    monkeypatch.setattr(benchmark, "_measure_once", fail_measure)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)
    tasks = {
        f"task-{index}": {"id": f"task-{index}", "prompt": "prompt"}
        for index in range(20)
    }

    results = benchmark.run_quality_suite(
        model="model",
        mode="warm",
        concurrency=2,
        tasks=tasks,
        repetitions=2,
    )

    assert len(calls) == 2
    assert len(results) == 2
    assert all(benchmark.COMPLETION_BUDGET_EXHAUSTED in result["error"] for result in results)


def test_trace_timeline_never_fabricates_missing_latency() -> None:
    assert benchmark.build_trace_timeline([], "any", None, None, None, None, None, None) == []


def test_trace_timeline_separates_measured_native_reasoning_without_fake_transition() -> None:
    steps = benchmark.build_trace_timeline(
        [{"retrieval_ms": 10.0}],
        "any",
        median_ttft_s=0.5,
        median_e2e_s=0.8,
        median_prefill_tps=100.0,
        median_decode_tps=20.0,
        prompt_tokens=50,
        output_tokens=10,
        median_generation_ttft_s=0.1,
        median_reasoning_s=0.4,
    )
    assert [step["step_type"] for step in steps] == [
        "retrieval",
        "llm_prefill",
        "llm_reasoning",
        "llm_decode",
    ]
    assert all(step["step_type"] != "transition" for step in steps)


def test_summary_uses_observed_context_and_median_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "model_metadata", lambda _: {})
    monkeypatch.setattr(benchmark, "runtime_metadata", lambda: {})
    runs = [
        {"run_index": 1, "error": None, "ttft_s": 0.1, "e2e_s": 1.0, "prefill_tps": 1.0,
         "decode_tps": 2.0, "model_memory_mb": None, "prompt_tokens": 10, "output_tokens": 20,
         "context_len": 4096},
        {"run_index": 1, "error": None, "ttft_s": 0.2, "e2e_s": 2.0, "prefill_tps": 2.0,
         "decode_tps": 3.0, "model_memory_mb": None, "prompt_tokens": 14, "output_tokens": 24,
         "context_len": 4096},
    ]

    summary = benchmark.summarise("model", "category", "warm", runs, engine="vllm")

    assert summary["context_len"] == 4096
    assert summary["prompt_tokens"] == 12
    assert summary["output_tokens"] == 22


def test_summary_does_not_turn_missing_usage_into_zero_aggregate_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "model_metadata", lambda _: {})
    runs = [{
        "run_index": 1,
        "burst_duration_s": 1.0,
        "ttft_s": 0.1,
        "e2e_s": 0.5,
        "prefill_tps": None,
        "decode_tps": None,
        "model_memory_mb": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "output_token_source": "unavailable",
    }]
    summary = benchmark.summarise("model", "suite", "warm", runs, engine="vllm")
    assert summary["aggregate_output_tps"] is None
    assert summary["quantization"] is None


def test_registered_category_runs_full_task_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every task supplied by the registry is executed."""
    from src import benchmark

    seen_task_ids = []

    def mock_measure(
        model,
        system,
        user,
        cold,
        run_index,
        req_index,
        task_id,
        store_response,
        min_ctx,
        sampling=None,
        response_format="text",
        max_output_tokens=None,
        generation_timeout_seconds=120,
    ):
        seen_task_ids.append(task_id)
        return {
            "run_index": run_index,
            "req_index": req_index,
            "cold": cold,
            "ttft_s": 0.1,
            "e2e_s": 0.5,
            "prefill_tps": 10.0,
            "decode_tps": 10.0,
            "prompt_tokens": 10,
            "output_tokens": 10,
            "load_duration_s": 0.0,
            "model_memory_mb": 1000,
            "memory_source": "unavailable",
            "response_preview": "test",
            "response": "",
            "task_id": task_id,
            "error": None,
        }

    monkeypatch.setattr(benchmark, "_measure_once", mock_measure)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)

    results = benchmark.run_benchmark(
        model="test",
        prompt_category="reasoning",
        mode="warm",
        num_runs=1,
        concurrency=1,
        custom_tasks=[
            {"id": "reasoning-1", "prompt": "one", "task_type": "numeric_exact", "expected_answer": "1"},
            {"id": "reasoning-2", "prompt": "two", "task_type": "numeric_exact", "expected_answer": "2"},
        ],
    )

    assert set(seen_task_ids) == {"reasoning-1", "reasoning-2"}
    assert len(results) == len(seen_task_ids)


def test_run_benchmark_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from src import benchmark

    call_timestamps = []

    def mock_measure(*args, **kwargs):
        call_timestamps.append(time.time())
        time.sleep(0.5)
        return {
            "run_index": kwargs.get("run_index", 1),
            "req_index": kwargs.get("req_index", 0),
            "cold": False,
            "ttft_s": 0.1,
            "e2e_s": 0.5,
            "prefill_tps": 10.0,
            "decode_tps": 10.0,
            "prompt_tokens": 10,
            "output_tokens": 10,
            "load_duration_s": 0.0,
            "model_memory_mb": 1000,
            "memory_source": "unavailable",
            "response_preview": "test",
            "response": "",
            "task_id": "test",
            "error": None
        }

    monkeypatch.setattr(benchmark, "_measure_once", mock_measure)
    monkeypatch.setattr(benchmark, "warm_model", lambda _: None)

    t0 = time.time()
    results = benchmark.run_benchmark(
        model="test",
        prompt_category="faq",
        mode="warm",
        num_runs=1,
        concurrency=3,
        custom_tasks=[
            {"id": f"task-{index}", "prompt": f"task {index}"}
            for index in range(3)
        ],
    )
    t1 = time.time()

    assert len(results) == 3
    # With stagger (0.5s * req_index) + 0.5s work per request:
    #   concurrent ≈ max stagger 1.0s + 0.5s work = 1.5s
    #   sequential ≈ (0 + 0.5 + 1.0) stagger + 3 * 0.5s work = 3.0s
    # 2.5s cleanly separates the two.
    assert (t1 - t0) < 2.5, "Execution was not concurrent"

    # Stagger deliberately offsets burst starts by 0.5s each, so starts must
    # NOT be simultaneous — but all three must still complete in parallel
    # (total < 2.5s, see above). Sequential execution would take ~3.0s.
    max_start_diff = max(call_timestamps) - min(call_timestamps)
    assert max_start_diff >= 0.4, "Stagger did not offset burst starts"

    assert all("burst_duration_s" in r for r in results)
    assert results[0]["burst_duration_s"] >= 0.5


def test_cold_ollama_rejects_concurrent_evictions() -> None:
    with pytest.raises(ValueError, match="concurrency=1"):
        benchmark.run_benchmark(
            model="test",
            prompt_category="unused",
            mode="cold",
            num_runs=1,
            concurrency=2,
            custom_tasks=[{"id": "test", "prompt": "test"}],
        )

def test_percentiles_exclude_failures() -> None:
    from src import benchmark

    def _make_mock(run_index, ttft_s, error):
        return {
            "run_index": run_index,
            "req_index": 0,
            "ttft_s": ttft_s,
            "e2e_s": ttft_s * 2 if ttft_s else None,
            "prefill_tps": 10.0,
            "decode_tps": 10.0,
            "model_memory_mb": 1000,
            "prompt_tokens": 10,
            "output_tokens": 10,
            "error": error
        }

    runs = [
        _make_mock(1, 1.0, None),
        _make_mock(1, 5.0, "timeout"),
        _make_mock(1, 2.0, None),
        _make_mock(1, None, "crashed"),
        _make_mock(1, 3.0, None),
    ]
    summary = benchmark.summarise("model", "faq", "warm", runs)
    assert summary["error_count"] == 2

    # 3 successes: 1.0, 2.0, 3.0. Median should be 2.0.
    # The failure with 5.0 should not skew the median or percentiles.
    assert summary["median_ttft_s"] == 2.0

    # Nearest-rank percentiles stay within the observed sample range.
    assert summary["p95_ttft_s"] == 3.0
    assert summary["p99_ttft_s"] == 3.0


def test_num_runs_pooling_behavior() -> None:
    from src import benchmark
    # num_runs=3, concurrency=5
    runs = []
    for i in range(1, 4):
        for j in range(5):
            runs.append({
                "run_index": i,
                "req_index": j,
                "output_tokens": 10,
                "prompt_tokens": 10,
                "burst_duration_s": 2.0,
                "error": None,
                "ttft_s": 1.0,
                "e2e_s": 2.0,
                "prefill_tps": 10.0,
                "decode_tps": 10.0,
                "model_memory_mb": 1000,
            })

    summary = benchmark.summarise("model", "faq", "warm", runs, concurrency=5)

    # We should detect 3 bursts
    assert summary["num_runs"] == 3
    # And concurrency 5 (explicit param — not inferred from run structure)
    assert summary["concurrency"] == 5

    # Each burst: 5 reqs * 10 tokens = 50 tokens. Duration = 2.0s.
    # Burst TPS = 50 / 2.0 = 25.0
    # Average across 3 bursts = 25.0
    assert summary["aggregate_output_tps"] == 25.0

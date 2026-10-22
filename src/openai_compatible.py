"""OpenAI-compatible hosted chat-completions benchmark adapter."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from src.benchmark import _stagger_delay
from src.benchmark_categories import validate_self_contained_tasks
from src.engines import (
    COMPLETION_BUDGET_EXHAUSTED,
    GENERATION_TIMEOUT_EXCEEDED,
    create_http_session,
    stream_request_timeout,
)

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
ABSOLUTE_MAX_CONCURRENCY = 128


def validate_provider_concurrency(concurrency: int) -> int:
    """Apply the runner's global safety bound without treating provider capacity as local VRAM."""
    if concurrency < 1:
        raise SystemExit(f"Error: --concurrency must be at least 1 (got {concurrency}).")
    if concurrency > ABSOLUTE_MAX_CONCURRENCY:
        raise SystemExit(
            f"Error: --concurrency {concurrency} exceeds the absolute limit of "
            f"{ABSOLUTE_MAX_CONCURRENCY}."
        )
    return concurrency


def normalize_base_url(value: str | None = None) -> str:
    """Normalize a credential-free OpenAI-compatible API base URL."""
    raw = (value if value is not None else os.environ.get(
        "OPENAI_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    )).strip().rstrip("/")
    parsed = urlsplit(raw)
    if not raw or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OPENAI_BASE_URL must not contain credentials, a query, or a fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def detect_openai_compatible(wait: bool = False) -> bool:
    """Check local configuration without making a billable provider request."""
    try:
        normalize_base_url()
    except ValueError:
        return False
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _empty_result(run_index: int, req_index: int, task_id: str, error: str) -> dict:
    return {
        "run_index": run_index,
        "req_index": req_index,
        "cold": False,
        "ttft_s": None,
        "generation_ttft_s": None,
        "reasoning_s": None,
        "e2e_s": None,
        "prefill_tps": None,
        "decode_tps": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "output_token_source": "unavailable",
        "load_duration_s": None,
        "model_memory_mb": None,
        "memory_source": "unavailable",
        "response_preview": "",
        "response": "",
        "task_id": task_id,
        "context_len": None,
        "error": error,
    }


def _stream_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item.get("text", "")
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _legacy_token_limit_required(response: requests.Response) -> bool:
    """Recognize compatibility endpoints that only accept the legacy field."""
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    detail = error.get("message") if isinstance(error, dict) else error
    message = str(detail or response.text or "").lower()
    return "max_completion_tokens" in message and any(
        marker in message
        for marker in ("unsupported", "unknown", "unrecognized", "not allowed", "extra field")
    )


def run_openai_compatible(
    model: str,
    prompt_category: str,
    mode: str,
    num_runs: int,
    concurrency: int = 1,
    store_responses: bool = False,
    min_ctx: int | None = None,
    sampling: dict | None = None,
    prompt_strategy: str | None = None,
    custom_system_prompt: str | None = None,
    max_output_tokens: int = 512,
    generation_timeout_seconds: int = 120,
    custom_tasks: list[dict] | None = None,
    reasoning_effort: str | None = None,
) -> list[dict]:
    """Run registered tasks through a hosted OpenAI-compatible chat API."""
    if mode != "warm":
        raise ValueError("OpenAI-compatible API benchmarks support only mode='warm'")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")
    if generation_timeout_seconds < 0:
        raise ValueError("generation_timeout_seconds must be at least 0")
    if not custom_tasks:
        raise ValueError("custom_tasks from the registered benchmark suite are required")

    validate_self_contained_tasks(custom_tasks)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the openai-compatible engine")
    base_url = normalize_base_url()
    session = create_http_session(
        pool_connections=max(10, concurrency * 2),
        pool_maxsize=max(10, concurrency * 2),
    )
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    tasks = custom_tasks

    def run_single(req_index: int, run_index: int, task: dict) -> dict:
        task_id = str(task["id"])
        from src.benchmark_categories import get_system_prompt_for_task

        system_prompt = custom_system_prompt or get_system_prompt_for_task(task, prompt_strategy)
        user_prompt = task.get("user", task.get("prompt", ""))
        _stagger_delay(req_index, concurrency)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": max_output_tokens,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["temperature"] = 0.0
        if sampling:
            payload.update(sampling)
        if reasoning_effort:
            # OpenAI reasoning models reject sampling controls such as
            # temperature and top_p when reasoning effort is enabled.
            payload.pop("temperature", None)
            payload.pop("top_p", None)

        request_started = time.perf_counter()
        first_content_at = None
        content_chunks: list[str] = []
        prompt_tokens = None
        output_tokens = None
        finish_reason = None
        stream_complete = False
        try:
            response = session.post(
                f"{base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=stream_request_timeout(generation_timeout_seconds, request_started),
            )
            if _legacy_token_limit_required(response):
                response.close()
                fallback_payload = dict(payload)
                fallback_payload["max_tokens"] = fallback_payload.pop("max_completion_tokens")
                response = session.post(
                    f"{base_url}/chat/completions",
                    json=fallback_payload,
                    stream=True,
                    timeout=stream_request_timeout(generation_timeout_seconds, request_started),
                )
            with response:
                if not response.ok:
                    # Do not include request headers or the response body in logs/results.
                    raise RuntimeError(f"provider returned HTTP {response.status_code}")
                for raw_line in response.iter_lines(chunk_size=None):
                    if not raw_line:
                        continue
                    if (
                        generation_timeout_seconds > 0
                        and time.perf_counter() - request_started >= generation_timeout_seconds
                    ):
                        raise RuntimeError(
                            f"{GENERATION_TIMEOUT_EXCEEDED} ({generation_timeout_seconds}s limit)"
                        )
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        stream_complete = True
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        p_count = usage.get("prompt_tokens")
                        o_count = usage.get("completion_tokens")
                        if isinstance(p_count, int) and p_count >= 0:
                            prompt_tokens = p_count
                        if isinstance(o_count, int) and o_count >= 0:
                            output_tokens = o_count
                    choices = chunk.get("choices") or []
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        text = _stream_content(delta.get("content"))
                        if text:
                            if first_content_at is None:
                                first_content_at = time.perf_counter()
                            content_chunks.append(text)
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]

            if not stream_complete:
                raise RuntimeError("provider stream ended before [DONE] was received")
            ended_at = time.perf_counter()
            answer = "".join(content_chunks)
            if finish_reason == "length":
                raise RuntimeError(f"{COMPLETION_BUDGET_EXHAUSTED} (requested={max_output_tokens})")
            if not answer.strip():
                raise RuntimeError("provider returned an empty assistant response")
            e2e_s = ended_at - request_started
            ttft_s = first_content_at - request_started if first_content_at else None
            decode_duration_s = ended_at - first_content_at if first_content_at else None
            decode_tps = (
                max(output_tokens - 1, 0) / decode_duration_s
                if output_tokens is not None and decode_duration_s and decode_duration_s > 0
                else None
            )
            total = f"/{len(tasks)}" if tasks else ""
            print(
                f"  [Run {run_index} | {req_index + 1}{total}] ✓ Task {task_id} | "
                f"TTFT: {ttft_s:.3f}s" if ttft_s is not None else
                f"  [Run {run_index} | {req_index + 1}{total}] ✓ Task {task_id} | TTFT: —",
                flush=True,
            )
            return {
                "run_index": run_index,
                "req_index": req_index,
                "cold": False,
                "ttft_s": ttft_s,
                "generation_ttft_s": ttft_s,
                "reasoning_s": None,
                "e2e_s": e2e_s,
                "prefill_tps": None,
                "decode_tps": decode_tps,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "output_token_source": "provider_usage" if output_tokens is not None else "unavailable",
                "load_duration_s": None,
                "model_memory_mb": None,
                "memory_source": "unavailable",
                "response_preview": answer[:120].replace("\n", " "),
                "response": answer if store_responses else "",
                "task_id": task_id,
                "context_len": None,
                "attempt_count": 1,
                "error": None,
            }
        except Exception as exc:
            if isinstance(exc, requests.exceptions.ReadTimeout) and generation_timeout_seconds > 0:
                error = f"{GENERATION_TIMEOUT_EXCEEDED} ({generation_timeout_seconds}s limit)"
            elif isinstance(exc, requests.exceptions.RequestException):
                error = f"provider request failed ({type(exc).__name__})"
            else:
                error = str(exc)
            result = _empty_result(run_index, req_index, task_id, error)
            return result

    results: list[dict] = []
    try:
        for run_index in range(1, num_runs + 1):
            burst_start = time.perf_counter()
            burst_results: list[dict] = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for batch_start in range(0, len(tasks), concurrency):
                    batch = tasks[batch_start:batch_start + concurrency]
                    futures = [
                        pool.submit(run_single, batch_start + offset, run_index, task)
                        for offset, task in enumerate(batch)
                    ]
                    batch_results = [future.result() for future in futures]
                    burst_results.extend(batch_results)
                    if burst_results and all(
                        COMPLETION_BUDGET_EXHAUSTED in str(row.get("error") or "").lower()
                        for row in burst_results
                    ):
                        break
            burst_duration = time.perf_counter() - burst_start
            for result in burst_results:
                result["burst_duration_s"] = burst_duration
                results.append(result)
            if burst_results and all(
                COMPLETION_BUDGET_EXHAUSTED in str(row.get("error") or "").lower()
                for row in burst_results
            ):
                break
        return results
    finally:
        session.close()

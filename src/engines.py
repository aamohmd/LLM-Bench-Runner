import os
import platform
import importlib.util
import time
import requests
from typing import TypedDict, Callable, Any
from urllib.parse import urlsplit, urlunsplit

from requests.adapters import HTTPAdapter

# Bounded stream timeout: (connect_timeout, read_timeout).
# 20s to connect, 120s max silence between tokens/chunks before timing out and auto-retrying.
STREAM_TIMEOUT: tuple[float, float] | float = (20.0, 120.0)
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
COMPLETION_BUDGET_EXHAUSTED = "completion budget exhausted before a complete assistant answer"
GENERATION_TIMEOUT_EXCEEDED = "generation timeout exceeded before a complete assistant answer"
ENGINE_UNAVAILABLE = "inference engine became unavailable during generation"


def normalize_vllm_api_base_url(value: str | None = None) -> str:
    """Return the canonical OpenAI-compatible vLLM API base ending in /v1."""
    raw = value if value is not None else os.environ.get("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)
    raw = raw.strip().rstrip("/")
    parsed = urlsplit(raw)
    if not raw or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("VLLM_BASE_URL must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("VLLM_BASE_URL must not include a query string or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _stream_text(value: object) -> str:
    """Return text from OpenAI-compatible string or typed content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _reasoning_delta(delta: dict) -> str:
    """Support current and legacy reasoning fields used by compatible servers."""
    for key in ("reasoning", "reasoning_content", "thinking"):
        text = _stream_text(delta.get(key))
        if text:
            return text
    return ""


def _stream_decode_tps(output_tokens: int | None, decode_duration_s: float | None) -> float | None:
    """Measure tokens after the first arrival over the post-TTFT interval."""
    if output_tokens is None or decode_duration_s is None or decode_duration_s <= 0:
        return None
    decoded_after_first = max(output_tokens - 1, 0)
    return decoded_after_first / decode_duration_s if decoded_after_first else None


def _completion_budget_failed(result: dict) -> bool:
    return COMPLETION_BUDGET_EXHAUSTED in str(result.get("error") or "").lower()


def _generation_timeout_failed(result: dict) -> bool:
    return GENERATION_TIMEOUT_EXCEEDED in str(result.get("error") or "").lower()


def _engine_unavailable_failed(result: dict) -> bool:
    return ENGINE_UNAVAILABLE in str(result.get("error") or "").lower()


def stream_request_timeout(
    generation_timeout_seconds: int,
    request_started_at: float,
) -> tuple[float, float] | float:
    """Bound a blocking stream read by the remaining generation deadline."""
    if generation_timeout_seconds <= 0:
        return STREAM_TIMEOUT
    if isinstance(STREAM_TIMEOUT, tuple):
        connect_timeout = float(STREAM_TIMEOUT[0])
        silence_timeout = float(STREAM_TIMEOUT[1])
    else:
        connect_timeout = float(STREAM_TIMEOUT)
        silence_timeout = float(STREAM_TIMEOUT)
    remaining = generation_timeout_seconds - (time.perf_counter() - request_started_at)
    return (connect_timeout, min(silence_timeout, max(remaining, 0.001)))


def _stream_read_timed_out(exc: Exception) -> bool:
    return isinstance(exc, requests.exceptions.ReadTimeout) or "read timed out" in str(exc).lower()


def _vllm_is_healthy(session: Any, base_url: str) -> bool:
    try:
        response = session.get(f"{base_url[:-3]}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def create_http_session(pool_connections: int = 100, pool_maxsize: int = 100) -> requests.Session:
    """Create a high-performance requests.Session with socket pooling and HTTP keep-alive."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=0,  # retries handled explicitly by post_with_retry
        pool_block=False,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Connection": "keep-alive"})
    return session


_GLOBAL_HTTP_SESSION = create_http_session(pool_connections=100, pool_maxsize=100)


def post_with_retry(
    session: Any = None,
    url: str = "",
    *,
    json: dict | None = None,
    stream: bool = True,
    timeout: tuple[float, float] | float = STREAM_TIMEOUT,
    retries: int = 2,
    backoff: float = 2.0,
) -> Any:
    """POST with a bounded timeout and bounded retries on transient failures.

    Retries on Timeout / ConnectionError / HTTP 5xx (server-side transient
    glitches). Other status codes (4xx etc.) and the final failure return
    immediately so the caller sees the real error via raise_for_status().
    """
    client = session if session is not None else _GLOBAL_HTTP_SESSION
    last_resp: Any = None
    for attempt in range(retries + 1):
        try:
            resp = client.post(url, json=json, stream=stream, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
        if resp.status_code < 500:
            return resp
        last_resp = resp
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    last_resp.raise_for_status()
    return last_resp

class EngineConfig(TypedDict):
    detect: Callable[..., bool]
    run: Callable[..., list[dict]]

def detect_ollama(wait: bool = False) -> bool:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False

def run_ollama(*args, **kwargs):
    from src.benchmark import run_benchmark
    return run_benchmark(*args, **kwargs)


def detect_openai_compatible(wait: bool = False) -> bool:
    from src.openai_compatible import detect_openai_compatible as detect
    return detect(wait=wait)


def run_openai_compatible(*args, **kwargs):
    from src.openai_compatible import run_openai_compatible as run
    return run(*args, **kwargs)


def detect_vllm(wait: bool = False) -> bool:
    import time
    try:
        base_url = normalize_vllm_api_base_url()
    except ValueError:
        return False
    root_url = base_url[:-3]
    # For large models (e.g. 70B+), downloading from HF can take >10 minutes depending on network throughput.
    # We increase max_retries to 120 (10 minutes) but recommend pre-populating the HF cache (see README).
    max_retries = 120 if wait else 1
    delay = 5 if wait else 0
    
    for _ in range(max_retries):
        try:
            version = requests.get(f"{root_url}/version", timeout=3)
            models = requests.get(f"{base_url}/models", timeout=3)
            if version.status_code == 200 and models.status_code == 200:
                return True
        except requests.RequestException:
            pass
        if wait:
            time.sleep(delay)
    return False

def run_vllm(
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
) -> list[dict]:
    import time
    import json
    from concurrent.futures import ThreadPoolExecutor
    from src.benchmark import (
        _stagger_delay,
        apply_context_window,
    )
    if mode != "warm":
        raise ValueError("vLLM benchmarks support only mode='warm'; cold mode requires restarting the server")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")
    if generation_timeout_seconds < 0:
        raise ValueError("generation_timeout_seconds must be at least 0")
    base_url = normalize_vllm_api_base_url()
    if not custom_tasks:
        raise ValueError("custom_tasks from the registered benchmark suite are required")
    from src.benchmark_categories import validate_self_contained_tasks
    validate_self_contained_tasks(custom_tasks)
    tasks_list = custom_tasks

    session = create_http_session(
        pool_connections=max(50, concurrency * 2),
        pool_maxsize=max(50, concurrency * 2),
    )

    # Validate the OpenAI-compatible route before launching the request pool and
    # dynamically discover the running server's max_model_len.
    server_max_model_len = None
    try:
        models_resp = session.get(f"{base_url}/models", timeout=5)
        models_resp.raise_for_status()
        m_list = models_resp.json().get("data", [])
        if not isinstance(m_list, list):
            raise ValueError("models response data is not a list")
        for m_item in m_list:
            if m_item.get("id") == model or not server_max_model_len:
                if "max_model_len" in m_item and m_item["max_model_len"]:
                    server_max_model_len = int(m_item["max_model_len"])
    except (requests.RequestException, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(
            f"vLLM preflight failed at {base_url}/models; configure the server root "
            "or its OpenAI API base ending in /v1"
        ) from exc

    reported_server_max_model_len = server_max_model_len
    effective_context_len = server_max_model_len or min_ctx or 4096

    # Dynamic chat template capability probe (Zero hardcoding of model names)
    # Probes the live vLLM endpoint with a 1-token dummy payload to test system role compatibility.
    model_supports_system_role = True
    try:
        probe_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "ping"},
                {"role": "user", "content": "ping"}
            ],
            "max_tokens": 1,
            "stream": False
        }
        probe_resp = session.post(f"{base_url}/chat/completions", json=probe_payload, timeout=10)
        if probe_resp.status_code == 400:
            probe_error = probe_resp.text.lower()
            if any(marker in probe_error for marker in (
                "system role", "roles must alternate", "conversation roles", "templateerror",
            )):
                model_supports_system_role = False
            else:
                probe_resp.raise_for_status()
        else:
            probe_resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"vLLM chat preflight failed at {base_url}/chat/completions"
        ) from exc

    try:
        def run_single(req_index: int, burst_index: int, custom_task: dict) -> dict:
            nonlocal model_supports_system_role
            current_task_id = custom_task["id"]
            from src.benchmark_categories import get_system_prompt_for_task
            current_sys = (
                custom_system_prompt
                if custom_system_prompt
                else get_system_prompt_for_task(custom_task, prompt_strategy)
            )
            current_user = custom_task.get("user", custom_task.get("prompt", ""))
            
            _stagger_delay(req_index, concurrency)
            
            # Guard against context window overflow by dynamically adapting max_tokens to server context limit
            # UTF-8 byte length is a conservative tokenizer-independent upper
            # bound that does not privilege Latin text over other scripts.
            approx_prompt_tokens = len(f"{current_sys or ''}{current_user or ''}".encode("utf-8")) + 32
            available_tokens = effective_context_len - approx_prompt_tokens - 16
            if available_tokens <= 0:
                return {
                    "run_index": burst_index,
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
                    "task_id": current_task_id,
                    "context_len": reported_server_max_model_len,
                    "error": (
                        "maximum context length exceeded: estimated prompt leaves no room "
                        f"for completion (context={effective_context_len}, prompt={approx_prompt_tokens})"
                    ),
                }
            safe_max_tokens = min(max_output_tokens, available_tokens)

            max_prompt_retries = 2
            request_start = time.perf_counter()
            for attempt in range(max_prompt_retries + 1):
                try:
                    if not model_supports_system_role:
                        # Dynamically detected that this model's chat template requires system prompts merged into user role.
                        messages = [
                            {"role": "user", "content": f"{current_sys}\n\n{current_user}" if current_sys else current_user}
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": current_sys},
                            {"role": "user", "content": current_user}
                        ]
                    payload: dict = {
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "temperature": 0.0,
                        "max_tokens": safe_max_tokens,
                        "seed": 42
                    }
                    if sampling:
                        payload.update(sampling)
                    # vLLM key_by_engine is None → apply_context_window is a no-op here;
                    # context window must be set at server launch (--max-model-len).
                    if min_ctx:
                        apply_context_window("vllm", payload, min_ctx)

                    full_response = []
                    reasoning_response = []
                    first_token_at = None
                    saw_reasoning = False
                    attempt_first_content_at = None
                    finish_reason = None
                    output_tokens = None
                    usage_output_tokens = None
                    output_token_source = "unavailable"
                    prompt_tokens = None
                    
                    # Request streaming from vLLM using pooled socket session
                    if (
                        attempt > 0
                        and generation_timeout_seconds > 0
                        and time.perf_counter() - request_start >= generation_timeout_seconds
                    ):
                        raise RuntimeError(
                            f"{GENERATION_TIMEOUT_EXCEEDED} "
                            f"({generation_timeout_seconds}s limit)"
                        )
                    with post_with_retry(
                        session,
                        f"{base_url}/chat/completions",
                        json=payload,
                        timeout=stream_request_timeout(
                            generation_timeout_seconds, request_start
                        ),
                        retries=0,
                    ) as resp:
                        resp.raise_for_status()
                        done = False
                        for raw_line in resp.iter_lines(chunk_size=None):
                            if not raw_line:
                                continue
                            if (
                                generation_timeout_seconds > 0
                                and time.perf_counter() - request_start >= generation_timeout_seconds
                            ):
                                raise RuntimeError(
                                    f"{GENERATION_TIMEOUT_EXCEEDED} "
                                    f"({generation_timeout_seconds}s limit)"
                                )
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                            if not line.startswith("data:"):
                                continue
                            
                            data_str = line[5:].lstrip()
                            if data_str.strip() == "[DONE]":
                                done = True
                                break
                                
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError as e:
                                raise RuntimeError(f"Malformed chunk from vLLM: {data_str}") from e
                            
                            choices = chunk.get("choices", [])
                            if choices:
                                choice = choices[0]
                                delta = choice.get("delta", {})
                                content = _stream_text(delta.get("content"))
                                reasoning_content = _reasoning_delta(delta)
                                if reasoning_content:
                                    if first_token_at is None:
                                        first_token_at = time.perf_counter()
                                    saw_reasoning = True
                                    reasoning_response.append(reasoning_content)
                                if content:
                                    if first_token_at is None:
                                        first_token_at = time.perf_counter()
                                    if attempt_first_content_at is None:
                                        attempt_first_content_at = time.perf_counter()
                                    full_response.append(content)
                                if choice.get("finish_reason") is not None:
                                    finish_reason = choice["finish_reason"]
                                    
                            # Some versions inject usage into the final chunk or a trailing chunk
                            usage = chunk.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens")
                                usage_output_tokens = usage.get("completion_tokens")
                                
                        if not done:
                            raise RuntimeError("vLLM stream dropped before [DONE] was received")

                    t_end = time.perf_counter()

                    e2e_s = t_end - request_start
                    generation_ttft_s = (first_token_at - request_start) if first_token_at else None
                    content_str = "".join(full_response)
                    reasoning_str = "".join(reasoning_response)
                    if finish_reason == "length":
                        answer_state = "before assistant content" if not content_str.strip() else "during assistant content"
                        raise RuntimeError(
                            f"{COMPLETION_BUDGET_EXHAUSTED} ({answer_state}; "
                            f"requested={safe_max_tokens}, available={available_tokens})"
                        )
                    if not content_str.strip():
                        if reasoning_str.strip():
                            raise RuntimeError(
                                "vLLM returned reasoning tokens but no assistant content "
                                f"(finish_reason={finish_reason or 'unknown'})"
                            )
                        raise RuntimeError(
                            "vLLM returned an empty assistant response "
                            f"(finish_reason={finish_reason or 'unknown'})"
                        )
                    ttft_s = (attempt_first_content_at - request_start) if attempt_first_content_at else None
                    reasoning_s = (
                        attempt_first_content_at - first_token_at
                        if saw_reasoning and first_token_at is not None and attempt_first_content_at is not None
                        else None
                    )
                    if isinstance(usage_output_tokens, int) and usage_output_tokens >= 0:
                        output_tokens = usage_output_tokens
                        output_token_source = "vllm_usage"
                    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
                        prompt_tokens = None

                    duration_s = (t_end - first_token_at) if first_token_at else e2e_s
                    decode_tps = _stream_decode_tps(output_tokens, duration_s)
                    prefill_tps = (
                        prompt_tokens / generation_ttft_s
                        if generation_ttft_s and generation_ttft_s > 0 and prompt_tokens
                        else None
                    )

                    ttft_str = f"{ttft_s:.3f}s" if ttft_s is not None else "—"
                    tps_str = f"{decode_tps:.1f} tok/s" if decode_tps is not None else "—"
                    out_str = f"{output_tokens} tok" if output_tokens is not None else "—"
                    total_str = f"/{len(tasks_list)}" if tasks_list else ""
                    print(f"  [Run {burst_index} | {req_index+1}{total_str}] ✓ Task {current_task_id} | TTFT: {ttft_str} | Speed: {tps_str} | Out: {out_str}", flush=True)

                    return {
                        "run_index": burst_index,
                        "req_index": req_index,
                        "cold": False,
                        "ttft_s": ttft_s,
                        "generation_ttft_s": generation_ttft_s,
                        "reasoning_s": reasoning_s,
                        "e2e_s": e2e_s,
                        "prefill_tps": prefill_tps,
                        "decode_tps": decode_tps,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "output_token_source": output_token_source,
                        "load_duration_s": None,
                        "model_memory_mb": None,
                        "memory_source": "unavailable",
                        "response_preview": content_str[:120].replace("\n", " "),
                        "response": content_str if store_responses else "",
                        "task_id": current_task_id,
                        "context_len": reported_server_max_model_len,
                        "attempt_count": attempt + 1,
                        "error": None
                    }
                except Exception as e:
                    if isinstance(
                        e,
                        (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
                    ):
                        if not _vllm_is_healthy(session, base_url):
                            e = RuntimeError(f"{ENGINE_UNAVAILABLE}: {e}")
                        elif generation_timeout_seconds > 0 and _stream_read_timed_out(e):
                            e = RuntimeError(
                                f"{GENERATION_TIMEOUT_EXCEEDED} "
                                f"({generation_timeout_seconds}s limit)"
                            )
                    err_body = ""
                    if isinstance(e, requests.exceptions.HTTPError) and getattr(e, "response", None) is not None:
                        try:
                            err_body = e.response.text or e.response.content.decode("utf-8", errors="replace")
                        except Exception:
                            pass
                    err_msg = f"{e} | Response: {err_body}" if err_body else str(e)
                    err_str = err_msg.lower()
                    is_template_role_error = any(kw in err_str for kw in [
                        "roles must alternate", "system role", "conversation roles", "templateerror", "does not support system"
                    ])
                    if is_template_role_error and attempt < max_prompt_retries:
                        model_supports_system_role = False
                        print(f"🔄 Model chat template requires merged system prompt. Dynamically adapting and retrying (attempt {attempt+1}/{max_prompt_retries})...")
                        continue

                    is_transient = any(t in err_str for t in [
                        "connection reset", "connection broken", "connection error",
                        "remotedisconnected", "read timed out", "timeout", "broken pipe",
                        "chunkedencodingerror"
                    ])
                    is_generation_timeout = GENERATION_TIMEOUT_EXCEEDED in err_str
                    is_engine_unavailable = ENGINE_UNAVAILABLE in err_str
                    if (
                        is_transient
                        and not is_generation_timeout
                        and not is_engine_unavailable
                        and attempt < max_prompt_retries
                    ):
                        backoff = 2.0 * (attempt + 1)
                        print(f"⚠️ vLLM run {burst_index}.{req_index} hit transient error ({e}). Retrying prompt in {backoff:.1f}s (attempt {attempt+1}/{max_prompt_retries})...")
                        time.sleep(backoff)
                        continue

                    total_str = f"/{len(tasks_list)}" if tasks_list else ""
                    print(f"  [Run {burst_index} | {req_index+1}{total_str}] ❌ Task {current_task_id} failed: {err_msg}", flush=True)
                    return {
                        "run_index": burst_index,
                        "req_index": req_index,
                        "cold": False,
                        "ttft_s": None,
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
                        "task_id": current_task_id,
                        "context_len": reported_server_max_model_len,
                        "error": err_msg
                    }

        results = []

        # Run the complete registered task list once per repetition. Submit
        # one concurrency-sized batch at a time so systematic failures cannot
        # waste the rest of a large category.
        for run_index in range(1, num_runs + 1):
            burst_start = time.perf_counter()
            burst_results = []
            timeout_circuit_open = False
            consecutive_full_batch_timeouts = 0
            systemic_timeout_threshold = max(2, min(concurrency, len(tasks_list)))
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for batch_start in range(0, len(tasks_list), concurrency):
                    batch = tasks_list[batch_start:batch_start + concurrency]
                    futures = [
                        pool.submit(run_single, batch_start + offset, run_index, task)
                        for offset, task in enumerate(batch)
                    ]
                    batch_results = [future.result() for future in futures]
                    burst_results.extend(batch_results)
                    if any(_engine_unavailable_failed(item) for item in batch_results):
                        print(
                            "  ■ vLLM became unavailable; remaining category tasks "
                            "were not started.",
                            flush=True,
                        )
                        timeout_circuit_open = True
                        break
                    batch_fully_timed_out = bool(batch_results) and all(
                        _generation_timeout_failed(item) for item in batch_results
                    )
                    consecutive_full_batch_timeouts = (
                        consecutive_full_batch_timeouts + len(batch_results)
                        if batch_fully_timed_out
                        else 0
                    )
                    if consecutive_full_batch_timeouts >= systemic_timeout_threshold:
                        print(
                            f"  ■ {consecutive_full_batch_timeouts} consecutive tasks across "
                            f"complete batches exceeded the {generation_timeout_seconds}s "
                            "generation limit; remaining category tasks were not started.",
                            flush=True,
                        )
                        timeout_circuit_open = True
                        break
                    if burst_results and all(_completion_budget_failed(item) for item in burst_results):
                        print(
                            f"  ■ Completion budget failed for the initial probe batch "
                            f"({len(batch_results)} tasks); remaining category tasks were not started.",
                            flush=True,
                        )
                        break
            burst_duration_s = time.perf_counter() - burst_start
            for result in burst_results:
                result["burst_duration_s"] = burst_duration_s
                results.append(result)
            if timeout_circuit_open:
                break
            if burst_results and all(_completion_budget_failed(item) for item in burst_results):
                break
        return results
    finally:
        try:
            session.close()
        except Exception:
            pass

ENGINES: dict[str, EngineConfig] = {
    "ollama": {"detect": detect_ollama, "run": run_ollama},
    "vllm": {"detect": detect_vllm, "run": run_vllm},
    "openai-compatible": {"detect": detect_openai_compatible, "run": run_openai_compatible},
}

#!/usr/bin/env python3
"""Shared inference measurement and summary functions for the remote runner."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid5

CONFIG_NAMESPACE = UUID("7f3b8b1a-9c42-4f2a-8b1e-0e9a7e2b5c4d")
import requests

from src.benchmark_categories import get_system_prompt_for_task, validate_self_contained_tasks
from src.engines import (
    COMPLETION_BUDGET_EXHAUSTED,
    GENERATION_TIMEOUT_EXCEEDED,
    post_with_retry,
    stream_request_timeout,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed generation settings (do not vary these between runs)
GENERATION_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 1024,  # max output tokens
}

# ---------------------------------------------------------------------------
# Concurrency guardrails
#
# The realistic risks from high concurrency differ by engine and don't scale
# the same way across hardware:
#   - vLLM: concurrent requests compete for KV-cache space bounded by
#     --max-model-len; too many at once surfaces as 400s or an OOM-killed
#     server, not a hardware fault (see the VRAM/KV-cache warning in the
#     project README).
#   - Ollama / CPU-bound engines: concurrency mostly saturates CPU threads
#     and memory bandwidth (or system unified memory), showing up
#     as request timeouts and skewed TTFT/decode numbers rather than a
#     crash.
#   - Any engine: hammering a shared/remote server other people depend on.
#
# Thresholds are computed from actual detected memory, not a lookup table of
# GB breakpoints — that way a 6GB card and a 141GB one both get a number
# computed from their real capacity instead of being snapped into buckets
# someone chose in advance, and nothing needs editing when a new GPU size
# ships. GPU VRAM is used when present (it's what bounds KV-cache for any
# GPU-accelerated engine); system RAM is used when there's no discrete GPU
# signal (CPU-only hosts, or unified memory architectures, where RAM *is*
# the real constraint). ABSOLUTE_MAX_CONCURRENCY is a flat sanity bound
# against typo/fat-fingered input, independent of hardware.
#
# MEMORY_GB_PER_CONCURRENT_REQUEST is the one real assumption left, and it's
# a deliberately conservative placeholder: actual memory per request depends
# on model size, quantization, context length, and max output tokens — none
# of which this module can know for an arbitrary model without querying that
# model's own config (layer count, hidden size, dtype). A more precise
# version would fetch that from the model's config.json (for vLLM/HF models)
# and compute real KV-cache bytes per token; this is the coarse, generic
# stand-in for hardware this code can't otherwise reason about.
# ---------------------------------------------------------------------------
ABSOLUTE_MAX_CONCURRENCY = 128
MEMORY_GB_PER_CONCURRENT_REQUEST = 3.0
STAGGER_SECONDS = 0.5


def _usable_memory_gb(hardware_info) -> tuple[Optional[float], str, str]:
    """Best-effort (usable_memory_gb, description, source) from a hardware_detect result.

    source is one of "gpu", "ram", or "none" — used downstream to gate
    GPU-specific advice (e.g. accelerator power limits) so it isn't shown for
    a RAM-based (no discrete GPU) reading. Prefers GPU VRAM (what bounds a
    GPU-accelerated engine's concurrency); falls back to system RAM when no
    discrete GPU is reported. Uses getattr defensively since the exact
    HardwareInfo shape lives in src/hardware_detect.py and this module
    doesn't own it.
    """
    if hardware_info is None:
        return None, "hardware not detected", "none"

    gpu_mb = getattr(hardware_info, "gpu_memory_mb", None)
    if gpu_mb:
        try:
            gpu_gb = float(gpu_mb) / 1024
            gpu_name = getattr(hardware_info, "gpu_name", None) or "detected GPU"
            return gpu_gb, f"{gpu_name} (~{gpu_gb:.0f}GB VRAM)", "gpu"
        except (TypeError, ValueError):
            pass

    ram_gb = getattr(hardware_info, "ram_gb", None)
    if ram_gb:
        try:
            ram_gb = float(ram_gb)
            return ram_gb, f"~{ram_gb:.0f}GB system RAM (no discrete GPU detected)", "ram"
        except (TypeError, ValueError):
            pass

    return None, "hardware not detected", "none"


def _hardware_concurrency_limits(hardware_info) -> tuple[int, int, str, str]:
    """Return (warn_threshold, hard_cap, description, source), computed from detected memory.

    No lookup table: both numbers are derived directly from usable_memory_gb
    via MEMORY_GB_PER_CONCURRENT_REQUEST, so they scale continuously with
    whatever hardware is actually reported. When nothing can be measured at
    all (detection failed, or returned neither GPU nor RAM info), falls back
    to the most conservative possible values rather than guessing.
    """
    usable_gb, label, source = _usable_memory_gb(hardware_info)
    if usable_gb is None:
        return 2, 4, label, source

    warn = max(2, int(usable_gb // MEMORY_GB_PER_CONCURRENT_REQUEST))
    # Hard cap allows roughly double the recommended budget as headroom
    # before refusing outright, capped by the flat absolute sanity bound.
    cap = max(warn + 1, int(usable_gb // (MEMORY_GB_PER_CONCURRENT_REQUEST / 2)))
    cap = min(cap, ABSOLUTE_MAX_CONCURRENCY)

    return warn, cap, f"{label}, ~{MEMORY_GB_PER_CONCURRENT_REQUEST:.0f}GB/request budget assumed", source


def validate_concurrency(
    concurrency: int,
    *,
    hardware_info=None,
    engine: str = "unknown",
    interactive: bool = True,
) -> int:
    """Enforce a hardware-scaled cap and, above a lower threshold, confirmation.

    Called by scripts/run_remote_benchmark.py before dispatching benchmark
    requests, so the maintained runner applies the same hardware guardrail.

    hardware_info, if provided, should be the result of detect_hardware() —
    used only to pick a sensible threshold; if it's
    None or detection failed, this falls back to the most conservative
    defaults (see _hardware_concurrency_limits).
    """
    if concurrency < 1:
        raise SystemExit(f"Error: --concurrency must be at least 1 (got {concurrency}).")

    if concurrency > ABSOLUTE_MAX_CONCURRENCY:
        raise SystemExit(
            f"Error: --concurrency {concurrency} exceeds the absolute limit of "
            f"{ABSOLUTE_MAX_CONCURRENCY}."
        )

    warn_threshold, hard_cap, hw_label, hw_source = _hardware_concurrency_limits(hardware_info)

    if concurrency > hard_cap:
        raise SystemExit(
            f"Error: --concurrency {concurrency} exceeds the cap of {hard_cap} for "
            f"{hw_label}. Pass a lower value, or check hardware detection if this "
            "looks wrong (e.g. GPU not visible to this process)."
        )

    if concurrency > warn_threshold and interactive:
        try:
            import questionary
        except ImportError:
            questionary = None

        reason = (
            "risking KV-cache exhaustion (400s / OOM) on vLLM"
            if engine == "vllm"
            else "risking CPU/memory contention and request timeouts"
            if engine == "ollama"
            else "risking resource contention"
        )
        message = (
            f"concurrency={concurrency} is above the recommended threshold of "
            f"{warn_threshold} for {hw_label} ({reason}). Continue?"
        )

        # Only relevant when a discrete GPU was actually the source of the
        # reading; meaningless for a RAM-based reading (no GPU) or when
        # hardware detection failed entirely.
        if hw_source == "gpu":
            print(
                "Tip: if you plan to run high-concurrency tests like this often, "
                "consider configuring hardware power capping limits on "
                "the machine running the engine, as a conservative default."
            )

        if questionary is not None:
            confirmed = questionary.confirm(message, default=False).ask()
            if not confirmed:
                raise SystemExit("Aborted by user.")
        else:
            print(f"Warning: {message} Proceeding non-interactively (questionary not installed).")

    return concurrency


def _stagger_delay(req_index: int, concurrency: int) -> None:
    """Sleep before an initial concurrent submission.

    Only meaningful for the first `concurrency`-many requests in a burst
    (req_index < concurrency): those are the ones ThreadPoolExecutor
    dispatches all at once. Anything queued beyond that only starts once a
    worker frees up — i.e. after a prior request naturally completes — so
    it's already spaced out and doesn't need an added delay.
    """
    if concurrency > 1 and STAGGER_SECONDS > 0 and req_index < concurrency:
        time.sleep(STAGGER_SECONDS * req_index)


def apply_context_window(engine: str, options: dict, min_ctx: int) -> dict:
    """Apply a minimum context-window requirement to an outgoing request options dict.

    Each engine uses a different key name; vLLM sets context at server launch
    (None key → no-op here, enforced by pre-flight check instead).
    Adding a new engine is one new entry in key_by_engine.
    min_ctx is read from the task suite's min_context_window field — changing
    it is a one-line JSON edit, not a code change.
    """
    key_by_engine: dict[str, str | None] = {
        "ollama": "num_ctx",
        "vllm": None,  # server-launch param only; see pre-flight verification
    }
    key = key_by_engine.get(engine)
    if key:
        options[key] = max(options.get(key, 0), min_ctx)
    return options

# ---------------------------------------------------------------------------
# Provider boundary: replace this client with another implementation for
# vLLM or a direct MLX runner without changing benchmark measurement logic.
# ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        from src.engines import create_http_session
        self._session = create_http_session(pool_connections=100, pool_maxsize=100)

    def get(self, path: str, **kwargs):
        url = path if path.startswith("http://") or path.startswith("https://") else f"{self.base_url}{path}"
        return self._session.get(url, **kwargs)

    def post(self, path: str, **kwargs):
        url = path if path.startswith("http://") or path.startswith("https://") else f"{self.base_url}{path}"
        return self._session.post(url, **kwargs)


CLIENT = OllamaClient(OLLAMA_BASE_URL)

def unload_model(model: str) -> None:
    """Ask Ollama to evict a model from memory (for cold runs)."""
    response = CLIENT.post(
        "/api/chat",
        json={"model": model, "messages": [], "keep_alive": 0},
        timeout=30,
    )
    response.raise_for_status()


def warm_model(model: str) -> None:
    """Load a model before a warm measurement without timing the load."""
    CLIENT.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "keep_alive": "5m",
            "options": {"num_predict": 1},
        },
        timeout=300,
    ).raise_for_status()


def runtime_metadata() -> dict:
    """Return the Ollama version when the endpoint is available."""
    try:
        response = CLIENT.get("/api/version", timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {}


def model_metadata(model: str) -> dict:
    """Return digest and quantization metadata for a local model."""
    metadata: dict = {}
    try:
        tags = CLIENT.get("/api/tags", timeout=10)
        tags.raise_for_status()
        for item in tags.json().get("models", []):
            if item.get("name") == model:
                metadata.update({"digest": item.get("digest"), "size_bytes": item.get("size")})
                break
    except (requests.RequestException, ValueError):
        pass

    try:
        details = CLIENT.post(
            "/api/show", json={"name": model}, timeout=30
        )
        details.raise_for_status()
        metadata.update(details.json().get("details", {}))
    except (requests.RequestException, ValueError):
        pass
    return metadata


def loaded_model_memory_mb(model: str) -> Optional[float]:
    """Read loaded model memory from Ollama's process list, when supported."""
    try:
        response = CLIENT.get("/api/ps", timeout=10)
        response.raise_for_status()
        for item in response.json().get("models", []):
            if item.get("name") == model:
                size = item.get("size_vram") or item.get("size")
                if size is not None:
                    return round(size / 1024 / 1024, 1)
    except (requests.RequestException, ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def _measure_once(
    model: str,
    system_prompt: str,
    user_prompt: str,
    cold: bool,
    run_index: int,
    req_index: int,
    task_id: str,
    store_response: bool = False,
    min_ctx: int | None = None,
    sampling: dict | None = None,
    max_output_tokens: int | None = None,
    generation_timeout_seconds: int = 120,
) -> dict:
    """
    Run a single inference and return a dict of raw metrics.

    Timing strategy:
      - Stream the response to capture TTFT accurately.
      - Parse the final stats block (sent as last JSON line by Ollama) for
        prompt_eval_count / prompt_eval_duration / eval_count / eval_duration.
      - Measure end-to-end wall-clock externally.
      - Sample loaded model memory from Ollama's /api/ps endpoint.
    """
    if cold:
        unload_model(model)
        time.sleep(2)  # let the eviction settle

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    gen_opts = dict(GENERATION_OPTIONS)
    if sampling:
        gen_opts.update(sampling)
    if max_output_tokens is not None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        gen_opts["num_predict"] = max_output_tokens
    if generation_timeout_seconds < 0:
        raise ValueError("generation_timeout_seconds must be at least 0")
    if min_ctx:
        apply_context_window("ollama", gen_opts, min_ctx)
    context_capacity = int(gen_opts.get("num_ctx") or 0)
    approx_prompt_tokens = len(f"{system_prompt}{user_prompt}".encode("utf-8")) + 32
    available_tokens = max(context_capacity - approx_prompt_tokens - 16, 1) if context_capacity else None
    if available_tokens is not None:
        gen_opts["num_predict"] = min(int(gen_opts["num_predict"]), available_tokens)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": gen_opts,
    }

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    t_first_content: Optional[float] = None
    full_response: list[str] = []
    reasoning_response: list[str] = []
    stats: dict = {}
    saw_reasoning = False

    initial_memory_mb = loaded_model_memory_mb(model)
    peak_model_memory_mb: Optional[float] = initial_memory_mb

    max_prompt_retries = 2
    for attempt in range(max_prompt_retries + 1):
        try:
            full_response = []
            reasoning_response = []
            stats = {}
            t_first_token = None
            t_first_content = None
            saw_reasoning = False
            saw_done = False
            if (
                attempt > 0
                and generation_timeout_seconds > 0
                and time.perf_counter() - t_start >= generation_timeout_seconds
            ):
                raise RuntimeError(
                    f"{GENERATION_TIMEOUT_EXCEEDED} "
                    f"({generation_timeout_seconds}s limit)"
                )
            with post_with_retry(
                CLIENT,
                "/api/chat",
                json=payload,
                timeout=stream_request_timeout(generation_timeout_seconds, t_start),
                retries=0,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines(chunk_size=None):
                    if not raw_line:
                        continue
                    if (
                        generation_timeout_seconds > 0
                        and time.perf_counter() - t_start >= generation_timeout_seconds
                    ):
                        raise RuntimeError(
                            f"{GENERATION_TIMEOUT_EXCEEDED} "
                            f"({generation_timeout_seconds}s limit)"
                        )
                    chunk = json.loads(raw_line)

                    # Track engine first-token time separately from user-visible
                    # time to first final-answer content. Native thinking remains
                    # enabled and its entire delay is charged to answer TTFT/E2E.
                    msg = chunk.get("message", {})
                    thinking = msg.get("thinking", "")
                    if t_first_token is None:
                        if msg.get("content") or thinking:
                            t_first_token = time.perf_counter()

                    if thinking:
                        saw_reasoning = True
                        reasoning_response.append(str(thinking))

                    content = msg.get("content", "")
                    if content:
                        if t_first_content is None:
                            t_first_content = time.perf_counter()
                        full_response.append(content)

                    # Final stats block
                    if chunk.get("done"):
                        stats = chunk
                        saw_done = True
                        break
            if not saw_done:
                raise RuntimeError("Ollama stream dropped before a done chunk was received")
            content_str = "".join(full_response)
            reasoning_str = "".join(reasoning_response)
            done_reason = str(stats.get("done_reason") or "unknown").lower()
            if done_reason == "length":
                answer_state = "before assistant content" if not content_str.strip() else "during assistant content"
                raise RuntimeError(
                    f"{COMPLETION_BUDGET_EXHAUSTED} ({answer_state}; "
                    f"requested={gen_opts['num_predict']}, available={available_tokens or gen_opts['num_predict']})"
                )
            if not content_str.strip():
                if reasoning_str.strip():
                    raise RuntimeError(
                        "Ollama returned reasoning tokens but no assistant content "
                        f"(done_reason={done_reason})"
                    )
                raise RuntimeError(
                    f"Ollama returned an empty assistant response (done_reason={done_reason})"
                )
            break
        except Exception as e:
            if generation_timeout_seconds > 0 and (
                isinstance(e, requests.exceptions.ReadTimeout)
                or "read timed out" in str(e).lower()
            ):
                e = RuntimeError(
                    f"{GENERATION_TIMEOUT_EXCEEDED} "
                    f"({generation_timeout_seconds}s limit)"
                )
            err_str = str(e).lower()
            is_transient = any(t in err_str for t in [
                "connection reset", "connection broken", "connection error",
                "remotedisconnected", "read timed out", "timeout", "broken pipe",
                "chunkedencodingerror"
            ])
            is_generation_timeout = GENERATION_TIMEOUT_EXCEEDED in err_str
            if is_transient and not is_generation_timeout and attempt < max_prompt_retries:
                backoff = 2.0 * (attempt + 1)
                time.sleep(backoff)
                continue
            raise e

    t_end = time.perf_counter()
    final_memory_mb = loaded_model_memory_mb(model)
    if final_memory_mb is not None:
        peak_model_memory_mb = max(peak_model_memory_mb or 0, final_memory_mb)

    # --- Derived metrics ---
    e2e_s = t_end - t_start
    generation_ttft_s = (t_first_token - t_start) if t_first_token else None
    ttft_s = (t_first_content - t_start) if t_first_content else None
    reasoning_s = (
        t_first_content - t_first_token
        if saw_reasoning and t_first_token is not None and t_first_content is not None
        else None
    )

    # Ollama reports durations in nanoseconds
    ns = 1e9
    prompt_eval_count = int(stats.get("prompt_eval_count") or 0)
    prompt_eval_ns = int(stats.get("prompt_eval_duration") or 0)
    eval_count = int(stats.get("eval_count") or 0)
    eval_ns = int(stats.get("eval_duration") or 0)
    load_ns = int(stats.get("load_duration") or 0)

    prefill_tps = (prompt_eval_count / (prompt_eval_ns / ns)) if prompt_eval_ns else None
    decode_tps = (eval_count / (eval_ns / ns)) if eval_ns else None

    return {
        "run_index": run_index,
        "req_index": req_index,
        "cold": cold,
        "ttft_s": round(ttft_s, 4) if ttft_s else None,
        "generation_ttft_s": round(generation_ttft_s, 4) if generation_ttft_s else None,
        "reasoning_s": round(reasoning_s, 4) if reasoning_s is not None else None,
        "e2e_s": round(e2e_s, 4),
        "prefill_tps": round(prefill_tps, 2) if prefill_tps else None,
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "prompt_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "output_token_source": "ollama_usage",
        "load_duration_s": round(load_ns / ns, 4) if load_ns else None,
        "model_memory_mb": peak_model_memory_mb,
        "memory_source": "ollama_api_ps" if peak_model_memory_mb is not None else "unavailable",
        "response_preview": content_str[:120].replace("\n", " "),
        "response": content_str if store_response else "",
        "task_id": task_id,
        "context_len": min_ctx,
        "attempt_count": attempt + 1,
        "error": None
    }


def run_benchmark(
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
    custom_tasks: dict[str, dict] | list[dict] | None = None,
) -> list[dict]:
    """
    Run `num_runs` repetitions and return raw per-run dicts.
    mode: 'warm' | 'cold'
    """
    cold = mode == "cold"
    if cold and concurrency != 1:
        raise ValueError("cold Ollama benchmarks require concurrency=1 so requests cannot evict each other")

    if not custom_tasks:
        raise ValueError("custom_tasks from the registered benchmark suite are required")
    tasks_dict = {t["id"]: t for t in custom_tasks} if isinstance(custom_tasks, list) else custom_tasks
    return run_quality_suite(
        model=model,
        mode=mode,
        concurrency=concurrency,
        tasks=tasks_dict,
        repetitions=num_runs,
        min_ctx=min_ctx,
        store_responses=store_responses,
        sampling=sampling,
        prompt_strategy=prompt_strategy,
        custom_system_prompt=custom_system_prompt,
        max_output_tokens=max_output_tokens,
        generation_timeout_seconds=generation_timeout_seconds,
    )

def run_quality_suite(
    model: str,
    mode: str,
    concurrency: int,
    tasks: dict,
    repetitions: int = 1,
    min_ctx: int | None = None,
    store_responses: bool = True,
    sampling: dict | None = None,
    prompt_strategy: str | None = None,
    custom_system_prompt: str | None = None,
    max_output_tokens: int = 512,
    generation_timeout_seconds: int = 120,
) -> list[dict]:
    """Execute every quality task once per requested repetition."""
    validate_self_contained_tasks(tasks)
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time
    from rich.console import Console
    console = Console()
    cold = mode == "cold"
    if cold and concurrency != 1:
        raise ValueError("cold Ollama quality benchmarks require concurrency=1")
    if not cold:
        warm_model(model)

    console.print(
        f"\n[bold cyan]Benchmarking Quality Suite[/bold cyan] [green]{model}[/green] "
        f"| tasks={len(tasks)} "
        f"| repetitions={repetitions} | mode=[magenta]{mode}[/magenta] | concurrency={concurrency}"
    )

    tasks_list = list(tasks.values())
    results = []
    progress_lock = Lock()
    completed_by_run = {run_index: 0 for run_index in range(1, repetitions + 1)}

    def report_task(run_index: int, task: dict, result: dict) -> None:
        with progress_lock:
            completed_by_run[run_index] += 1
            completed = completed_by_run[run_index]
            prefix = (
                f"  Pass {run_index}/{repetitions} · " if repetitions > 1 else "  "
            )
            if result.get("error"):
                print(
                    f"{prefix}Task {completed}/{len(tasks_list)} ✗ {task['id']} | "
                    f"{result['error']}",
                    flush=True,
                )
                return
            ttft = f"{result['ttft_s']:.3f}s" if result.get("ttft_s") is not None else "—"
            speed = (
                f"{result['decode_tps']:.1f} tok/s"
                if result.get("decode_tps") is not None
                else "—"
            )
            output = (
                f"{result['output_tokens']} tok"
                if result.get("output_tokens") is not None
                else "—"
            )
            print(
                f"{prefix}Task {completed}/{len(tasks_list)} ✓ {task['id']} | "
                f"TTFT {ttft} · {speed} · {output}",
                flush=True,
            )

    def run_single(req_index: int, run_index: int, task: dict) -> dict:
        _stagger_delay(req_index, concurrency)
        try:
            system_prompt = custom_system_prompt or get_system_prompt_for_task(task, prompt_strategy)
            user_prompt = task.get("user", task.get("prompt", ""))
            
            res = _measure_once(
                model,
                system_prompt,
                user_prompt,
                cold=cold,
                run_index=run_index,
                req_index=req_index,
                task_id=task["id"],
                store_response=store_responses,
                min_ctx=min_ctx,
                sampling=sampling,
                max_output_tokens=max_output_tokens,
                generation_timeout_seconds=generation_timeout_seconds,
            )
            report_task(run_index, task, res)
            return res
        except Exception as exc:
            result = {
                "run_index": run_index,
                "req_index": req_index,
                "cold": cold,
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
                "task_id": task["id"],
                "error": str(exc)
            }
            report_task(run_index, task, result)
            return result

    for run_index in range(1, repetitions + 1):
        burst_start = time.perf_counter()
        console.print(f"  Pass {run_index}/{repetitions}: {len(tasks_list)} tasks (concurrency={concurrency})…")
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
                batch_fully_timed_out = bool(batch_results) and all(
                    GENERATION_TIMEOUT_EXCEEDED
                    in str(result.get("error") or "").lower()
                    for result in batch_results
                )
                consecutive_full_batch_timeouts = (
                    consecutive_full_batch_timeouts + len(batch_results)
                    if batch_fully_timed_out
                    else 0
                )
                if consecutive_full_batch_timeouts >= systemic_timeout_threshold:
                    console.print(
                        f" [red]{consecutive_full_batch_timeouts} consecutive tasks across "
                        f"complete batches exceeded the {generation_timeout_seconds}s generation "
                        "limit; remaining tasks were not started.[/red]"
                    )
                    timeout_circuit_open = True
                    break
                if burst_results and all(
                    COMPLETION_BUDGET_EXHAUSTED in str(result.get("error") or "").lower()
                    for result in burst_results
                ):
                    console.print(
                        f" [red]Completion budget failed for the initial {len(batch_results)}-task "
                        "probe; remaining tasks were not started.[/red]"
                    )
                    break
        burst_duration_s = time.perf_counter() - burst_start
        for result in burst_results:
            result["burst_duration_s"] = burst_duration_s
            results.append(result)
        if timeout_circuit_open:
            break
        successes = [result for result in burst_results if not result.get("error")]
        if successes:
            console.print(f" completed in {burst_duration_s:.2f}s ({len(successes)} successful, {len(burst_results) - len(successes)} failed)")
        else:
            console.print(" [red]All tasks failed.[/red]")
        if burst_results and all(
            COMPLETION_BUDGET_EXHAUSTED in str(result.get("error") or "").lower()
            for result in burst_results
        ):
            break

    return results



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _median_or_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _median_int_or_none(values: list) -> Optional[int]:
    clean = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    return int(round(statistics.median(clean))) if clean else None

def _percentile_or_none(values: list, p: float) -> Optional[float]:
    clean = sorted([v for v in values if v is not None])
    if not clean:
        return None
    if not 0 < p <= 100:
        raise ValueError("percentile must be in the interval (0, 100]")
    # Nearest-rank is bounded by the observed sample range, unlike the
    # default exclusive quantile estimator which extrapolates for small N.
    import math
    return clean[max(0, math.ceil((p / 100) * len(clean)) - 1)]


def derive_config_identity(config_dict: dict) -> tuple[str, str]:
    """Derives deterministic UUIDv5 and SHA-256 hash from canonicalized configuration dictionary.

    Excludes None values, sorts keys strictly, and dumps with compact separators
    to guarantee identical hashes across platforms and serialization invocations.
    """
    clean_dict = {k: v for k, v in config_dict.items() if v is not None}
    canonical_str = json.dumps(clean_dict, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    config_id = str(uuid5(CONFIG_NAMESPACE, config_hash))
    return config_id, config_hash


def build_trace_timeline(
    runs: list[dict],
    prompt_category: str,
    median_ttft_s: Optional[float],
    median_e2e_s: Optional[float],
    median_prefill_tps: Optional[float],
    median_decode_tps: Optional[float],
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
    median_generation_ttft_s: Optional[float] = None,
    median_reasoning_s: Optional[float] = None,
) -> list[dict]:
    """Construct a waterfall containing measured stages only."""
    steps = []
    cur_ms = 0.0

    retrieval_ms = None
    runs_with_retrieval = [r.get("retrieval_ms") for r in runs if r.get("retrieval_ms") is not None]
    if runs_with_retrieval:
        retrieval_ms = float(statistics.median(runs_with_retrieval))
    if retrieval_ms is not None:
        steps.append({
            "step_name": "Vector Retrieval",
            "step_type": "retrieval",
            "start_ms": round(cur_ms, 2),
            "duration_ms": round(retrieval_ms, 2),
            "metadata": {},
        })
        cur_ms += retrieval_ms

    prefill_s = median_generation_ttft_s
    if prefill_s is None and median_reasoning_s is None:
        prefill_s = median_ttft_s
    if prefill_s is not None:
        ttft_ms = prefill_s * 1000.0
        steps.append({
            "step_name": "LLM Prefill (Generation TTFT)",
            "step_type": "llm_prefill",
            "start_ms": round(cur_ms, 2),
            "duration_ms": round(ttft_ms, 2),
            "metadata": {
                "prompt_tokens": prompt_tokens,
                "prefill_tps": median_prefill_tps,
            },
        })
        cur_ms += ttft_ms

    if median_reasoning_s is not None and median_reasoning_s > 0:
        reasoning_ms = median_reasoning_s * 1000.0
        steps.append({
            "step_name": "Native Model Reasoning",
            "step_type": "llm_reasoning",
            "start_ms": round(cur_ms, 2),
            "duration_ms": round(reasoning_ms, 2),
            "metadata": {},
        })
        cur_ms += reasoning_ms

    if median_e2e_s is not None and median_ttft_s is not None and median_e2e_s >= median_ttft_s:
        decode_ms = (median_e2e_s - median_ttft_s) * 1000.0
    elif median_decode_tps and median_decode_tps > 0 and output_tokens:
        decode_ms = (output_tokens / median_decode_tps) * 1000.0
    else:
        decode_ms = None

    if decode_ms is not None:
        steps.append({
            "step_name": "LLM Autoregressive Generation",
            "step_type": "llm_decode",
            "start_ms": round(cur_ms, 2),
            "duration_ms": round(decode_ms, 2),
            "metadata": {
                "output_tokens": output_tokens,
                "decode_tps": median_decode_tps,
            },
        })

    return steps


def summarise(
    model: str,
    prompt_category: str,
    mode: str,
    runs: list[dict],
    context_len: Optional[int] = None,
    engine: str = "ollama",
    concurrency: int = 1,
    pipeline_config_id: Optional[str] = None,
    trace_timeline: Optional[list[dict]] = None,
    quantization: Optional[str] = None,
    model_revision: Optional[str] = None,
    engine_config: Optional[dict] = None,
) -> dict:
    """Compute summary row from raw runs. Percentiles and metrics only use successful runs."""
    metadata = model_metadata(model) if engine == "ollama" else {}
    runtime = runtime_metadata() if engine == "ollama" else {}
    
    success_runs = [r for r in runs if not r.get("error")]
    error_count = len(runs) - len(success_runs)
    generation_timeout_count = sum(
        GENERATION_TIMEOUT_EXCEEDED in str(run.get("error") or "").lower()
        for run in runs
    )
    generation_timeout_rate = generation_timeout_count / len(runs) if runs else 0.0
    
    ttft_values = [r["ttft_s"] for r in success_runs if r.get("ttft_s") is not None]
    p95_val = _percentile_or_none(ttft_values, 95) if ttft_values else None
    p99_val = _percentile_or_none(ttft_values, 99) if ttft_values else None

    aggregate_output_tps = None
    has_authoritative_burst_tokens = success_runs and all(
        isinstance(run.get("output_tokens"), int)
        and not isinstance(run.get("output_tokens"), bool)
        and run.get("output_tokens") >= 0
        and isinstance(run.get("burst_duration_s"), (int, float))
        for run in success_runs
    )
    if has_authoritative_burst_tokens:
        bursts = {}
        for r in success_runs:
            b_idx = r.get("run_index")
            if b_idx not in bursts:
                bursts[b_idx] = {"duration": r["burst_duration_s"], "tokens": 0}
            bursts[b_idx]["tokens"] += r["output_tokens"]
        
        tps_list = [b["tokens"] / b["duration"] for b in bursts.values() if b["duration"] > 0]
        if tps_list:
            aggregate_output_tps = round(sum(tps_list) / len(tps_list), 2)

    token_sources = {r.get("output_token_source") for r in success_runs if r.get("output_token_source")}
    output_token_source = next(iter(token_sources)) if len(token_sources) == 1 else ("mixed" if token_sources else "unavailable")

    inferred_context_lengths = [
        r.get("context_len") for r in success_runs
        if isinstance(r.get("context_len"), int) and r.get("context_len") > 0
    ]
    if context_len is None and inferred_context_lengths:
        context_len = int(statistics.median(inferred_context_lengths))

    # Robust git revision detection
    git_rev = os.environ.get("GIT_REVISION")
    if not git_rev:
        try:
            git_rev = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            revision_file = Path(__file__).resolve().parent.parent / "REVISION"
            try:
                git_rev = revision_file.read_text(encoding="utf-8").strip() or None
            except OSError:
                git_rev = None

    # Quantization is provenance, so never invent a value from a model name or
    # engine default. Ollama exposes it as metadata; other engines must supply it.
    quant_val = quantization
    if quant_val is None and engine == "ollama":
        quant_val = metadata.get("quantization")

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt_category": prompt_category,
        "mode": mode,
        "context_len": context_len,
        "num_runs": len(set(r.get("run_index", i) for i, r in enumerate(runs))),
        "concurrency": concurrency,
        "error_count": error_count,
        "generation_timeout_count": generation_timeout_count,
        "generation_timeout_rate": generation_timeout_rate,
        "median_ttft_s": _median_or_none([r["ttft_s"] for r in success_runs]),
        "median_generation_ttft_s": _median_or_none([
            r.get("generation_ttft_s") for r in success_runs
        ]),
        "median_reasoning_s": _median_or_none([r.get("reasoning_s") for r in success_runs]),
        "p95_ttft_s": round(p95_val, 4) if p95_val is not None else None,
        "p99_ttft_s": round(p99_val, 4) if p99_val is not None else None,
        "median_e2e_s": _median_or_none([r["e2e_s"] for r in success_runs]),
        "median_prefill_tps": _median_or_none([r["prefill_tps"] for r in success_runs]),
        "median_decode_tps": _median_or_none([r["decode_tps"] for r in success_runs]),
        "aggregate_output_tps": aggregate_output_tps,
        "median_model_memory_mb": _median_or_none([r["model_memory_mb"] for r in success_runs]),
        "prompt_tokens": _median_int_or_none([r.get("prompt_tokens") for r in success_runs]),
        "output_tokens": _median_int_or_none([r.get("output_tokens") for r in success_runs]),
        "output_token_source": output_token_source,
        "framework": engine,
        "quantization": quant_val,
        "git_revision": git_rev,
        "model_revision": model_revision or (metadata.get("digest") if engine == "ollama" else None),
        "hardware": os.environ.get("BENCHMARK_HARDWARE", platform.platform()),
        "os": sys.platform,
    }

    if engine == "ollama":
        result.update({
            "model_digest": metadata.get("digest"),
            "model_size_mb": round(metadata["size_bytes"] / 1024 / 1024, 1) if metadata.get("size_bytes") else None,
            "ollama_version": runtime.get("version"),
        })

    # Deterministic configuration identity derivation
    cfg_identity_dict = {
        "engine": str(engine).lower().strip(),
        "model": str(model).strip(),
        "quantization": quant_val,
        "max_context_len": context_len,
        "gpu_memory_utilization": None,
        "retrieval_k": None,
        "reranker_model": None,
        "agent_topology": "direct",
        "parameters": engine_config or {},
    }
    derived_cfg_id, _ = derive_config_identity(cfg_identity_dict)
    active_pipeline_config_id = pipeline_config_id or derived_cfg_id

    active_trace_timeline = trace_timeline if trace_timeline is not None else build_trace_timeline(
        runs=success_runs,
        prompt_category=prompt_category,
        median_ttft_s=result["median_ttft_s"],
        median_e2e_s=result["median_e2e_s"],
        median_prefill_tps=result["median_prefill_tps"],
        median_decode_tps=result["median_decode_tps"],
        prompt_tokens=result["prompt_tokens"],
        output_tokens=result["output_tokens"],
        median_generation_ttft_s=result["median_generation_ttft_s"],
        median_reasoning_s=result["median_reasoning_s"],
    )

    result.update({
        "pipeline_config_id": active_pipeline_config_id,
        "trace_timeline": active_trace_timeline,
        "engine_config": engine_config or {},
    })

    return result


def save_run_csv(model: str, prompt_category: str, mode: str, runs: list[dict]) -> Path:
    tag = model.replace(":", "_").replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_task_tag = runs[0].get("task_id", prompt_category) if runs else prompt_category
    task_tag = str(raw_task_tag).replace("/", "_").replace(":", "_")
    path = RESULTS_DIR / f"{tag}_{task_tag}_{mode}_{ts}.csv"
    if not runs:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=runs[0].keys())
        writer.writeheader()
        writer.writerows(runs)
    return path

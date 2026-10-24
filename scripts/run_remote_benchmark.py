#!/usr/bin/env python3
"""
Interactive CLI for running local benchmarks and posting results to a remote backend.
Execution-plane script run on GPU nodes to submit data back to control plane.
Collaborates with src/benchmark.py and src/hardware_detect.py.
Falls back to local JSON save if the backend network request fails.
"""

import argparse
from contextlib import ExitStack
import math
import sys
import os
import requests
import json
import subprocess
import hashlib
from pathlib import Path
from uuid import UUID, uuid4
from urllib.parse import urlsplit
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): pass

try:
    import questionary
except ImportError:
    questionary = None

def _check_interactive():
    if not sys.stdin.isatty():
        print("Error: This command needs more arguments. Supply the missing CLI flags; interactive prompts require a terminal.", file=sys.stderr)
        raise SystemExit(2)
    if questionary is None:
        print("Error: Missing required CLI arguments, and 'questionary' is not installed for interactive prompts.")
        print("Please provide all CLI flags (e.g. model, --engine, --suite, --backend-url, --api-key) or install questionary: pip install questionary")
        sys.exit(1)


def _ask(prompt):
    value = prompt.ask()
    if value is None:
        print("Cancelled. No further benchmark tasks were started.", file=sys.stderr)
        raise SystemExit(130)
    return value


def _validate_backend_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Backend URL must be an http:// or https:// address without credentials, query parameters or fragments")
    _ = parsed.port
    return value

# Load only this checkout's configuration before initializing engine clients.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmark import save_run_csv, summarise, validate_concurrency
from src.engines import COMPLETION_BUDGET_EXHAUSTED, GENERATION_TIMEOUT_EXCEEDED
from src.hardware_detect import detect_hardware, detect_hardware_from_endpoint


RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
INGESTION_CONTRACT_VERSION = 4
MAX_SUBMISSION_ERROR_RATE = 0.05


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return parsed


def _submission_error_rate_exceeded(
    error_count: int,
    total_requests: int,
    threshold: float = MAX_SUBMISSION_ERROR_RATE,
) -> bool:
    if total_requests <= 0:
        return True
    return error_count / total_requests > threshold


def _submission_blocked_by_task_errors(
    errors: list[object],
    total_requests: int,
    threshold: float = MAX_SUBMISSION_ERROR_RATE,
) -> bool:
    """Treat model generation limits as failed outcomes, while guarding corrupt runs."""
    infrastructure_error_count = sum(
        not any(
            marker in str(error).lower()
            for marker in (GENERATION_TIMEOUT_EXCEEDED, COMPLETION_BUDGET_EXHAUSTED)
        )
        for error in errors
    )
    return _submission_error_rate_exceeded(
        infrastructure_error_count,
        total_requests,
        threshold,
    )


def _missing_category_rows(
    expected: list[str],
    submitted: list[tuple[str, object]],
    saved: list[tuple[str, object]],
) -> list[str]:
    covered = {category for category, _ in [*submitted, *saved]}
    return [category for category in expected if category not in covered]


def _response_error_details(response: requests.Response | None) -> tuple[str, str, str | None]:
    """Return bounded, safe control-plane diagnostics from an HTTP response."""
    if response is None:
        return "http_error", "No HTTP response was returned.", None
    error_code = f"http_{response.status_code}"
    error_id = response.headers.get("X-Error-ID")
    message = "No diagnostic detail was returned."
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error_code = str(body.get("error_code") or error_code)
        error_id = str(body.get("error_id") or error_id or "") or None
        detail = body.get("detail")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail")
        if detail:
            message = str(detail)
    elif response.text and not response.text.lstrip().startswith("<"):
        message = response.text
    message = " ".join(message.split())[:500]
    return error_code, message, error_id


def _check_submission_readiness(backend_url: str, api_key: str) -> dict:
    """Fail before inference when result ingestion is incompatible or unavailable."""
    url = f"{backend_url.rstrip('/')}/api/v1/benchmark-results/readiness"
    try:
        response = requests.get(url, headers={"X-API-Key": api_key}, timeout=15, allow_redirects=False)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Cannot verify result ingestion ({type(exc).__name__}). "
            "No benchmark tasks were started."
        ) from exc

    if response.status_code == 404:
        raise RuntimeError(
            "The control plane is running an older ingestion contract. Deploy the current "
            "backend before running the runner; no benchmark tasks were started."
        )
    if not response.ok:
        error_code, message, error_id = _response_error_details(response)
        reference = f" · error {error_id}" if error_id else ""
        raise RuntimeError(
            f"Result ingestion preflight failed · HTTP {response.status_code} · "
            f"{error_code}{reference} · {message} No benchmark tasks were started."
        )
    try:
        readiness = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Result ingestion preflight returned invalid JSON; no benchmark tasks were started."
        ) from exc
    if not isinstance(readiness, dict):
        raise RuntimeError(
            "Result ingestion preflight returned an invalid document; no benchmark tasks were started."
        )
    contract_version = readiness.get("contract_version")
    if readiness.get("status") != "ready" or contract_version != INGESTION_CONTRACT_VERSION:
        raise RuntimeError(
            f"Unsupported result ingestion contract {contract_version!r}; expected "
            f"{INGESTION_CONTRACT_VERSION}. No benchmark tasks were started."
        )
    revision = readiness.get("git_revision") or "revision unavailable"
    print(
        f"✓ Result ingestion ready · contract {contract_version} · {revision}",
        flush=True,
    )
    return readiness


def _stable_task_subset(tasks: list[dict], limit: int | None) -> list[dict]:
    """Select a reproducible registry-order-independent tuning subset."""
    if limit is None or limit >= len(tasks):
        return list(tasks)
    return sorted(
        tasks,
        key=lambda task: hashlib.sha256(str(task["id"]).encode("utf-8")).hexdigest(),
    )[:limit]


def _select_benchmark_groups(
    groups: dict[str, list[dict]], only_categories: list[str] | None
) -> dict[str, list[dict]]:
    """Select explicitly requested registered categories for partial resume."""
    if not only_categories:
        return groups
    requested = set(only_categories)
    unknown = requested - set(groups)
    if unknown:
        raise ValueError(f"unknown benchmark categories: {sorted(unknown)}")
    return {
        category_id: tasks
        for category_id, tasks in groups.items()
        if category_id in requested
    }


def _validate_benchmark_tasks(tasks: list[dict]) -> None:
    """Fail before inference when a registry suite cannot be scored reliably."""
    from benchmark_scoring import validate_task

    if not tasks:
        raise ValueError("benchmark suite has no tasks")
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task at index {index} has no non-empty id")
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        prompt = task.get("prompt", task.get("user"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id!r} has no non-empty prompt")
        try:
            validate_task(task)
        except ValueError as exc:
            raise ValueError(f"task {task_id!r}: {exc}") from exc

def get_ollama_models():
    """Run `ollama list` and parse the model names."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True, timeout=10)
        lines = result.stdout.strip().split("\n")[1:] # Skip header
        models = [line.split()[0] for line in lines if line]
        return models
    except Exception:
        return []

def fetch_backend_suites(backend_url: str, api_key: str | None = None) -> list[dict]:
    """Fetch registered benchmark suites from the control plane."""
    try:
        headers = {"X-API-Key": api_key} if api_key else {}
        resp = requests.get(f"{backend_url.rstrip('/')}/api/v1/suites", headers=headers, timeout=5, allow_redirects=False)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Note: Unable to fetch suites from the control plane ({e}).")
    return []


def fetch_backend_prompts(backend_url: str, api_key: str) -> list[dict]:
    """Fetch active, admin-authorized prompt versions from the control plane."""
    try:
        response = requests.get(
            f"{backend_url.rstrip('/')}/api/v1/prompts",
            headers={"X-API-Key": api_key}, timeout=5, allow_redirects=False,
        )
        if response.ok:
            return response.json()
        print(f"Warning: prompt registry returned HTTP {response.status_code}.")
    except Exception as exc:
        print(f"Warning: unable to fetch prompt registry ({exc}).")
    return []


def fetch_backend_strategies(backend_url: str, api_key: str) -> list[dict]:
    """Fetch active prompt strategies from the control plane."""
    try:
        response = requests.get(
            f"{backend_url.rstrip('/')}/api/v1/strategies",
            headers={"X-API-Key": api_key},
            timeout=5,
            allow_redirects=False,
        )
        if response.ok:
            payload = response.json()
            return [
                strategy for strategy in payload
                if isinstance(strategy, dict) and strategy.get("name") and strategy.get("is_active", True)
            ]
        print(f"Warning: strategy registry returned HTTP {response.status_code}.")
    except Exception as exc:
        print(f"Warning: unable to fetch strategy registry ({exc}).")
    return []


def fetch_backend_suite_tasks(backend_url: str, suite_id: str, api_key: str | None = None) -> list[dict]:
    """Fetch task definitions for a specific suite from the control plane."""
    try:
        headers = {"X-API-Key": api_key} if api_key else {}
        resp = requests.get(f"{backend_url.rstrip('/')}/api/v1/suites/{suite_id}/tasks", headers=headers, timeout=15, allow_redirects=False)
        if resp.status_code == 200:
            tasks = resp.json()
            # Ensure any nested scoring_config keys are unpacked at task top level
            unpacked_tasks = []
            for t in tasks:
                td = dict(t)
                cfg = td.get("scoring_config")
                if cfg and isinstance(cfg, dict):
                    for k, v in cfg.items():
                        if k not in td:
                            td[k] = v
                unpacked_tasks.append(td)
            return unpacked_tasks
        else:
            print(f"⚠️ Warning: Backend returned HTTP {resp.status_code} for suite '{suite_id}' tasks: {resp.text[:200]}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to connect to backend for suite '{suite_id}' tasks: {e}")
    return []


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Durably replace a JSON outbox file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{uuid4().hex}.tmp"
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems cannot sync directory handles. The file itself
            # is still flushed and atomically replaced.
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_submission_id(payload: dict) -> bool:
    """Add a durable idempotency identity, returning whether payload changed."""
    existing = payload.get("submission_id")
    if existing is None:
        payload["submission_id"] = str(uuid4())
        return True
    try:
        payload["submission_id"] = str(UUID(str(existing)))
    except ValueError as exc:
        raise ValueError("fallback payload has an invalid submission_id") from exc
    return payload["submission_id"] != existing


def _save_to_outbox(payload: dict) -> Path:
    """Persist a submission before network I/O and return its stable path."""
    _ensure_submission_id(payload)
    path = RESULTS_DIR / f"fallback_{payload['submission_id']}.json"
    _atomic_write_json(path, payload)
    return path


def _find_legacy_submission(
    payload: dict,
    backend_url: str,
    api_key: str,
    timeout_seconds: int,
) -> dict | None:
    """Resolve a pre-v4 fallback by its unique campaign/category identity."""
    experiment_tag = payload.get("experiment_tag")
    if not experiment_tag:
        raise ValueError(
            "legacy fallback has no submission_id or experiment_tag; refusing unsafe replay"
        )
    response = requests.get(
        f"{backend_url.rstrip('/')}/api/v1/benchmark-history",
        params={"limit": 1000, "experiment_tag": experiment_tag},
        headers={"X-API-Key": api_key},
        timeout=(10, timeout_seconds),
        allow_redirects=False,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError("benchmark history returned an invalid legacy-replay response")

    identity_fields = (
        "model",
        "prompt_category",
        "mode",
        "concurrency",
        "framework",
        "pipeline_config_id",
        "prompt_strategy",
        "quantization",
        "model_revision",
    )
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and all(
            payload.get(field) is None or row.get(field) == payload.get(field)
            for field in identity_fields
        )
    ]
    if len(matches) > 1:
        raise ValueError(
            "legacy fallback matches multiple history rows; refusing ambiguous replay"
        )
    return matches[0] if matches else None


def _post_outbox_file(
    path: Path,
    backend_url: str,
    api_key: str,
    timeout_seconds: int,
) -> tuple[dict, dict]:
    """Remove a durable entry only after a valid stored-run acknowledgement."""
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("submission_id") is None:
        existing = _find_legacy_submission(
            payload, backend_url, api_key, timeout_seconds
        )
        if existing is not None:
            if not isinstance(existing.get("id"), str) or not existing["id"].strip():
                raise ValueError("Legacy reconciliation returned no stored run ID; keeping the outbox file")
            path.unlink(missing_ok=True)
            return payload, {"status": "already_created", "id": existing.get("id")}
    if _ensure_submission_id(payload):
        _atomic_write_json(path, payload)

    response = requests.post(
        f"{backend_url.rstrip('/')}/api/v1/benchmark-results",
        json=payload,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        timeout=(10, timeout_seconds),
        allow_redirects=False,
    )
    response.raise_for_status()
    try:
        acknowledgement = response.json()
    except ValueError as exc:
        raise requests.RequestException("Submission returned no valid storage acknowledgement; keeping the outbox file") from exc
    if (not 200 <= response.status_code < 300 or not isinstance(acknowledgement, dict)
            or acknowledgement.get("status") not in {"created", "already_created"}
            or not isinstance(acknowledgement.get("id"), str) or not acknowledgement["id"].strip()):
        raise requests.RequestException("Submission did not confirm a stored run ID; keeping the outbox file")

    path.unlink(missing_ok=True)
    return payload, acknowledgement


def _replay_fallbacks(
    backend_url: str,
    api_key: str,
    timeout_seconds: int = 120,
) -> tuple[list[tuple[dict, dict]], int]:
    """Drain durable fallback files, retaining every unacknowledged entry."""
    fallbacks = sorted(RESULTS_DIR.glob("fallback_*.json"))
    if not fallbacks:
        print("No fallback files to replay.")
        return [], 0

    submitted: list[tuple[dict, dict]] = []
    fail = 0
    for path in fallbacks:
        print(f"\n→ Replaying {path.name} ...")
        try:
            payload, replay_ack = _post_outbox_file(
                path, backend_url, api_key, timeout_seconds
            )
            print(
                f"  model={payload.get('model', '?')}  "
                f"peak_gpu_mem_mb={payload.get('peak_gpu_mem_mb')}  "
                f"sample_count={payload.get('sample_count')}"
            )
            print(f"  ✓ acknowledged → id={replay_ack.get('id', 'acknowledged')}")
            print("  removed acknowledged local outbox file")
            submitted.append((payload, replay_ack))
        except requests.HTTPError as e:
            error_code, message, error_id = _response_error_details(e.response)
            reference = f" · error {error_id}" if error_id else ""
            print(
                f"  ✗ HTTP {e.response.status_code} · {error_code}{reference} · {message}"
            )
            fail += 1
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            fail += 1
    print(f"\nDone: {len(submitted)} OK, {fail} failed.")
    return submitted, fail


def _main(resources):
    
    # We still keep argparse for users who want to run it non-interactively
    parser = argparse.ArgumentParser(description="Run benchmark and submit to remote control plane")
    parser.add_argument("--backend-url", help="URL of the remote control plane")
    parser.add_argument("--api-key", help="Runner API key")
    parser.add_argument("model", nargs="?", help="Model to benchmark")
    parser.add_argument("--concurrency", type=_positive_int, help="Concurrent requests")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "ollama", "openai-compatible"],
        help="Inference engine or hosted API adapter to use",
    )
    parser.add_argument("--suite", type=str, help="Benchmark suite ID or name to run (fetched from backend)")
    parser.add_argument("--category", type=str, help="Prompt category (legacy alias for suite)")
    parser.add_argument("--mode", type=str, choices=["warm", "cold"], help="Mode (warm or cold)")
    parser.add_argument("--runs", type=_positive_int, help="Number of repetitions")
    parser.add_argument("--capture-traces", action="store_true", help="Capture raw text responses for failed tasks (Audit Log)")
    parser.add_argument("--decision-mode", action="store_true", help="Run the quality suite 3x and report score + range (min/max/stdev) instead of a point estimate. Only valid for scored categories; cannot be combined with --runs.")
    parser.add_argument("--skip-concurrency-check", action="store_true", help="Bypass the VRAM-based concurrency cap (use when engine is a remote server).")
    parser.add_argument("--experiment-tag", type=str, help="Required unique, descriptive tag for grouping real runs")
    parser.add_argument("--pipeline-config-id", type=str, help="Deterministic UUIDv5 or custom pipeline configuration identifier")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Registered foundational prompt name (required for non-interactive runs)",
    )
    parser.add_argument(
        "--strategy",
        "--prompt-strategy",
        dest="prompt_strategy",
        type=str,
        default=None,
        help="Registered prompt directive strategy (required for non-interactive runs)",
    )
    parser.add_argument("--max-output-tokens", type=_positive_int, default=512, help="Maximum generated tokens per request")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Reasoning effort for the OpenAI-compatible engine; omits unsupported sampling controls",
    )
    parser.add_argument(
        "--generation-timeout-seconds",
        type=_non_negative_int,
        default=120,
        help="Maximum wall-clock generation time per task; 0 disables it (default: 120)",
    )
    parser.add_argument(
        "--tasks-per-category",
        type=_positive_int,
        default=None,
        help="Deterministic number of registry tasks per category (default: complete category)",
    )
    parser.add_argument(
        "--only-category",
        dest="only_categories",
        action="append",
        default=None,
        help="Run only this registered category; repeat for partial campaign resume.",
    )
    parser.add_argument("--quantization", type=str, default=None, help="Exact runtime quantization/dtype provenance")
    parser.add_argument("--model-revision", type=str, default=None, help="Exact model revision, digest, or commit provenance")
    parser.add_argument("--run-type", choices=("diagnostic", "approval"), default="diagnostic")
    parser.add_argument("--concurrency-scope", choices=("exploratory", "required"), default="exploratory")
    parser.add_argument("--approval-run-id", type=str, default=None, help="Approval campaign identifier")
    parser.add_argument(
        "--submission-timeout-seconds",
        type=_positive_int,
        default=120,
        help="Maximum time to wait for a result-ingestion response (default: 120)",
    )
    parser.add_argument(
        "--replay-fallbacks",
        action="store_true",
        help="Replay any fallback_*.json files in data/results/ that were saved during a backend outage.",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=None,
        help=(
            "vLLM server root or OpenAI API base ending in /v1 "
            "(both forms are normalized)"
        ),
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (defaults to OPENAI_BASE_URL or the OpenAI API)",
    )

    parser.add_argument("--ollama-base-url", help="Ollama server root URL reachable from this runner")
    parser.add_argument("--temperature", type=float, default=None, help="Request sampling temperature (0..2); default 0")
    parser.add_argument("--seed", type=_non_negative_int, default=None, help="Request seed; default 42 (not a determinism guarantee)")
    parser.add_argument("--response-format", choices=("text", "json"), default="text", help="JSON mode requires JSON-compatible tasks and references")
    parser.add_argument("--context-length", type=_positive_int, help="Ollama request context window; automatic when omitted")
    parser.add_argument("--launch-vllm", action="store_true", help="Start a local vLLM process for this benchmark and stop it on exit; requires an unused explicit --vllm-base-url")
    parser.add_argument("--vllm-max-model-len", type=_positive_int, help="Managed vLLM server context limit")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, help="Managed vLLM GPU memory fraction, greater than 0 and at most 1")
    parser.add_argument("--vllm-quantization", help="Managed vLLM quantization method supported by your installed version/model")
    parser.add_argument("--vllm-kv-cache-dtype", help="Managed vLLM KV cache dtype supported by your installed version/hardware")
    parser.add_argument("--vllm-prefix-caching", choices=("enabled", "disabled"), help="Managed vLLM prefix cache; omitted uses engine default")
    parser.add_argument("--vllm-startup-timeout", type=_positive_int, default=600, help="Managed vLLM startup deadline in seconds")

    args = parser.parse_args()

    if args.engine == "ollama" and os.environ.get("OLLAMA_NUM_PARALLEL"):
        try:
            _positive_int(os.environ["OLLAMA_NUM_PARALLEL"])
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("OLLAMA_NUM_PARALLEL must be a positive whole number")

    if args.temperature is not None and (not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2):
        parser.error("--temperature must be a finite value between 0 and 2")
    memory_fraction = args.vllm_gpu_memory_utilization
    if memory_fraction is not None and (not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1):
        parser.error("--vllm-gpu-memory-utilization must be greater than 0 and at most 1")
    startup_options = {
        "max_model_len": args.vllm_max_model_len,
        "gpu_memory_utilization": memory_fraction,
        "quantization": args.vllm_quantization,
        "kv_cache_dtype": args.vllm_kv_cache_dtype,
        "prefix_caching": args.vllm_prefix_caching,
    }
    if any(value is not None for value in startup_options.values()) and not args.launch_vllm:
        parser.error("vLLM startup settings require --launch-vllm; they cannot change an existing server")
    if args.launch_vllm and (args.engine != "vllm" or not args.vllm_base_url):
        parser.error("--launch-vllm requires --engine vllm and an explicit unused local --vllm-base-url")
    if args.context_length is not None and args.engine != "ollama":
        parser.error("--context-length is Ollama-only; use --vllm-max-model-len with --launch-vllm")
    if args.decision_mode and (args.temperature is not None or args.seed is not None):
        parser.error("decision mode defines its own temperature and seeds; omit --temperature and --seed")
    if args.engine == "openai-compatible" and (args.temperature is not None or args.seed is not None or args.response_format != "text"):
        parser.error("these request controls currently support vLLM and Ollama only")
    if args.vllm_quantization and args.quantization and args.vllm_quantization != args.quantization:
        parser.error("--quantization provenance must match --vllm-quantization")

    if args.ollama_base_url:
        parsed_ollama = urlsplit(args.ollama_base_url)
        if (parsed_ollama.scheme not in {"http", "https"} or not parsed_ollama.hostname
                or parsed_ollama.username or parsed_ollama.password or parsed_ollama.query
                or parsed_ollama.fragment or parsed_ollama.path.rstrip("/")):
            parser.error("--ollama-base-url must be an HTTP server root without credentials, path, query, or fragment")
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url.rstrip("/")
        from src import benchmark
        benchmark.OLLAMA_BASE_URL = args.ollama_base_url.rstrip("/")
        benchmark.CLIENT.base_url = benchmark.OLLAMA_BASE_URL
    if args.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_base_url.rstrip("/")
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")

    # Load from env if not provided
    backend_url = (
        args.backend_url
        or os.environ.get("BENCHMARK_API_URL")
        or os.environ.get("BACKEND_URL")
    )
    api_key = args.api_key or os.environ.get("API_KEY")
    if backend_url:
        try:
            backend_url = _validate_backend_url(backend_url)
        except ValueError as exc:
            parser.error(str(exc))

    if args.replay_fallbacks:
        if not backend_url:
            _check_interactive()
            backend_url = _ask(questionary.text("Enter Backend URL:"))
        if not api_key:
            _check_interactive()
            api_key = _ask(questionary.password("Enter API Key:"))
        try:
            backend_url = _validate_backend_url(backend_url)
        except ValueError as exc:
            parser.error(str(exc))
        try:
            _check_submission_readiness(backend_url, api_key)
        except RuntimeError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            sys.exit(1)
        _, replay_failures = _replay_fallbacks(
            backend_url, api_key, args.submission_timeout_seconds
        )
        if replay_failures:
            raise SystemExit(4)
        return

    if not args.experiment_tag or not args.experiment_tag.strip():
        parser.error(
            "--experiment-tag is required for every real benchmark run; "
            "use a unique descriptive value such as purpose-hardware-model-date"
        )
    if args.run_type == "approval":
        if not args.approval_run_id:
            parser.error("--approval-run-id is required for --run-type approval")
        if args.concurrency_scope != "required":
            parser.error("approval runs require --concurrency-scope required")
        if not args.model_revision or not args.quantization:
            parser.error("approval runs require exact --model-revision and --quantization provenance")

    if not backend_url:
        _check_interactive()
        backend_url = _ask(questionary.text("Enter Backend URL:"))
    if not api_key:
        _check_interactive()
        api_key = _ask(questionary.password("Enter API Key:"))

    if not backend_url or not api_key:
        print("Backend URL and API Key are required.")
        sys.exit(1)

    try:
        backend_url = _validate_backend_url(backend_url)
    except ValueError as exc:
        parser.error(str(exc))

    from benchmark_scoring.registry import fingerprint
    try:
        response = requests.get(f"{backend_url.rstrip('/')}/api/v1/scorers/catalog", headers={"X-API-Key": api_key}, timeout=15, allow_redirects=False)
        response.raise_for_status()
        catalog = response.json()
        if not isinstance(catalog, dict):
            parser.error("Platform scorer catalog is not a JSON object. No benchmark tasks were started.")
        if catalog.get("fingerprint") != fingerprint():
            parser.error("Platform and runner scoring implementations differ. Install matching releases before benchmarking.")
    except (requests.RequestException, ValueError) as exc:
        parser.error(f"Cannot verify the platform scoring catalog: {exc}")

    submission_contract_ready = True
    try:
        _check_submission_readiness(backend_url, api_key)
    except RuntimeError as exc:
        submission_contract_ready = False
        print(f"⚠️ {exc}", file=sys.stderr)
        print(
            "  Benchmark execution will continue; completed categories will be saved for replay.",
            file=sys.stderr,
        )

    registered_prompts = fetch_backend_prompts(backend_url, api_key)
    registered_strategies = fetch_backend_strategies(backend_url, api_key)
    target_prompt_name = args.prompt
    if not target_prompt_name and sys.stdin.isatty() and registered_prompts:
        _check_interactive()
        target_prompt_name = _ask(questionary.select(
            "Select foundational prompt:",
            choices=[prompt["name"] for prompt in registered_prompts],
        ))
    if not target_prompt_name:
        parser.error(
            "--prompt is required for non-interactive benchmark runs; "
            "choose an active prompt from the control-plane registry"
        )
    target_strategy = args.prompt_strategy
    if not target_strategy and sys.stdin.isatty() and registered_strategies:
        _check_interactive()
        target_strategy = _ask(questionary.select(
            "Select prompt strategy:",
            choices=[strategy["name"] for strategy in registered_strategies],
        ))
    if not target_strategy:
        parser.error(
            "--strategy is required for non-interactive benchmark runs; "
            "choose an active strategy from the control-plane registry"
        )

    selected_prompt = None
    # Try dynamic composition endpoint on control plane
    try:
        resp = requests.get(
            f"{backend_url.rstrip('/')}/api/v1/prompts/compose",
            params={"prompt": target_prompt_name, "strategy": target_strategy},
            headers={"X-API-Key": api_key},
            timeout=5,
            allow_redirects=False,
        )
        if resp.ok:
            comp_data = resp.json()
            selected_prompt = {
                "id": comp_data["base"]["id"],
                "name": comp_data["base"]["name"],
                "content_hash": comp_data["composite_hash"],
                "system_prompt": comp_data["system_prompt"],
                "prompt_strategy": (comp_data.get("strategy") or {}).get("name") or target_strategy,
            }

    except Exception:
        pass

    if not selected_prompt:
        parser.error(
            f"unable to resolve registered prompt {target_prompt_name!r} with strategy "
            f"{target_strategy!r}; refusing to run with mislabeled prompt provenance"
        )

    engine = args.engine
    if not engine:
        _check_interactive()
        engine = _ask(questionary.select(
            "Select inference engine:",
            choices=["ollama", "vllm", "openai-compatible"]
        ))

    if args.openai_reasoning_effort and engine != "openai-compatible":
        parser.error("--openai-reasoning-effort requires --engine openai-compatible")
    openai_request_options = (
        {"reasoning_effort": args.openai_reasoning_effort}
        if engine == "openai-compatible" and args.openai_reasoning_effort
        else {}
    )

    model = args.model
    if not model:
        _check_interactive()
        if engine == "ollama":
            available_models = get_ollama_models()
            if available_models:
                model = _ask(questionary.select(
                    "Select model to benchmark:",
                    choices=available_models + ["Other (type manually)"]
                ))
                
                if model == "Other (type manually)":
                    model = _ask(questionary.text("Enter model name:"))
            else:
                model = _ask(questionary.text("Enter the model name reported by the live engine:"))
        else:
            model = _ask(questionary.text("Enter the model name reported by the live engine:"))

    if not model:
        print("Model is required.")
        sys.exit(1)

    if engine == "openai-compatible" and not os.environ.get("OPENAI_API_KEY", "").strip():
        if sys.stdin.isatty():
            _check_interactive()
            provider_key = _ask(questionary.password("Enter OpenAI-compatible provider API key:"))
            if provider_key:
                os.environ["OPENAI_API_KEY"] = provider_key
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            parser.error(
                "--engine openai-compatible requires OPENAI_API_KEY in the environment "
                "or an interactive terminal"
            )

    # Suite / Category selection
    selected_suite = None
    selected_suite_id = args.suite or args.category
    suite_name = None
    backend_suites = fetch_backend_suites(backend_url, api_key)

    if not selected_suite_id:
        _check_interactive()
        if backend_suites:
            choices = []
            suite_map = {}
            for s in backend_suites:
                label = f"{s.get('name', s['id'])} ({s.get('task_count', 0)} tasks, domain: {s.get('domain', 'general')}) [id: {s['id']}]"
                choices.append(label)
                suite_map[label] = s
            choices.append("Legacy category / Other...")
            selected_choice = _ask(questionary.select("Select benchmark suite:", choices=choices))
            if selected_choice in suite_map:
                selected_suite = suite_map[selected_choice]
                selected_suite_id = selected_suite["id"]
                suite_name = selected_suite.get("name")
            else:
                selected_suite_id = _ask(questionary.text("Enter benchmark suite or category ID:"))
        else:
            selected_suite_id = _ask(questionary.text("Enter benchmark suite or category ID:"))

    category = selected_suite_id
    if selected_suite is None and backend_suites:
        for s in backend_suites:
            if s["id"].lower() == selected_suite_id.lower() or s["name"].lower() == selected_suite_id.lower():
                selected_suite = s
                selected_suite_id = s["id"]
                suite_name = s.get("name")
                break

    # Detect hardware now so the concurrency guardrail below can scale to it;
    # reused later for the submitted payload instead of detecting twice.
    if engine == "openai-compatible":
        hardware_info = None
    else:
        try:
            hw_url = os.environ.get(
                "VLLM_HARDWARE_URL" if engine == "vllm" else "OLLAMA_HARDWARE_URL"
            )
            base_engine_url = os.environ.get("OLLAMA_BASE_URL") if engine == "ollama" else os.environ.get("VLLM_BASE_URL")
            is_remote = False
            if base_engine_url:
                parsed_u = urlsplit(base_engine_url)
                if parsed_u.hostname and parsed_u.hostname not in ("127.0.0.1", "localhost", "0.0.0.0"):
                    is_remote = True
                    if not hw_url:
                        hw_url = f"{parsed_u.scheme or 'http'}://{parsed_u.hostname}:8081"

            if is_remote or hw_url:
                hardware_info = None
                if hw_url:
                    import time
                    for attempt in range(4):
                        hardware_info = detect_hardware_from_endpoint(hw_url)
                        if hardware_info and hardware_info.gpu_name:
                            break
                        time.sleep(1.0)
                if not hardware_info or not hardware_info.gpu_name:
                    print(f"Warning: remote hardware detection failed at {hw_url} after retries. Recording status=failed (will not fall back to local host).")
                    hardware_info = None
            else:
                hardware_info = detect_hardware()
        except Exception as exc:
            print(f"Warning: hardware detection failed ({exc}); using conservative concurrency defaults.")
            hardware_info = None

    concurrency_str = str(args.concurrency) if args.concurrency else ""
    if not concurrency_str:
        _check_interactive()
        concurrency_str = _ask(questionary.text(
            "Enter concurrency level:",
            default="1"
        ))
    try:
        concurrency = _positive_int(concurrency_str)
    except (ValueError, argparse.ArgumentTypeError):
        parser.error("Concurrency must be a positive whole number")

    if engine == "openai-compatible":
        from src.openai_compatible import validate_provider_concurrency
        concurrency = validate_provider_concurrency(concurrency)
        print(
            "Hosted API concurrency is sent as configured; check the provider's rate limits.",
            flush=True,
        )
    elif not args.skip_concurrency_check:
        is_interactive = sys.stdin.isatty() and not bool(args.concurrency)
        concurrency = validate_concurrency(
            concurrency,
            hardware_info=hardware_info,
            engine=engine,
            interactive=is_interactive,
        )
    if engine == "ollama":
        server_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
        if server_parallel and server_parallel.isdigit():
            concurrency = min(concurrency, int(server_parallel))

    mode = args.mode
    if not mode:
        _check_interactive()
        mode = _ask(questionary.select(
            "Select mode:",
            choices=["warm", "cold"]
        ))
    if engine in {"vllm", "openai-compatible"} and mode == "cold":
        parser.error(f"{engine} benchmarks support only --mode warm")
        
    runs_str = str(args.runs) if args.runs else ""
    if args.decision_mode:
        if args.runs and args.runs != 3:
            parser.error("--decision-mode forces 3x repetitions; cannot combine with --runs.")
        runs = 3
    elif not runs_str:
        _check_interactive()
        runs_str = _ask(questionary.text(
            "Enter number of repetitions (runs):",
            default="1"
        ))
        try:
            runs = _positive_int(runs_str)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("Repetitions must be a positive whole number")
    else:
        runs = args.runs
    
    print(f"\n---", flush=True)
    print(f"Running benchmark for {model} on {engine} with concurrency {concurrency}...", flush=True)
    print(f"Category: {category} | Mode: {mode} | Runs: {runs}", flush=True)
    print(f"---\n", flush=True)
    
    try:
        from src.engines import ENGINES
        if engine not in ENGINES:
            raise ValueError(f"Unknown engine: '{engine}'. Valid engines are: {list(ENGINES.keys())}")
        runner = ENGINES[engine]["run"]
            
        # Determine upfront if this category or suite has a quality evaluator / task set.
        # Remote benchmarks execute exclusively against canonical tasks in the backend database.
        # Local task file access is not permitted.
        custom_tasks = None
        quality_tasks = None
        if not selected_suite_id:
            print(
                f"\n❌ FATAL: No suite ID specified.\n"
                f"   Remote benchmarks must target a registered suite in the backend database.\n"
                f"   Use --suite <id> or select a suite interactively.\n",
                flush=True,
            )
            sys.exit(1)

        fetched_tasks = fetch_backend_suite_tasks(backend_url, selected_suite_id, api_key)
        if not fetched_tasks:
            print(
                f"\n❌ FATAL: Failed to load tasks for suite '{selected_suite_id}' from backend ({backend_url}).\n"
                f"   Remote benchmarks must execute against canonical tasks from the backend database.\n"
                f"   Local task file access is not permitted.\n",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        try:
            _validate_benchmark_tasks(fetched_tasks)
        except ValueError as exc:
            parser.error(f"invalid benchmark suite {selected_suite_id!r}: {exc}")

        print(f"✓ Loaded {len(fetched_tasks)} tasks from backend database for suite '{selected_suite_id}'.", flush=True)

        # Partition tasks into individual benchmark workloads
        benchmark_groups: dict[str, list[dict]] = {}
        for t in fetched_tasks:
            cat_id = t.get("category_id") or selected_suite_id
            benchmark_groups.setdefault(cat_id, []).append(t)
        try:
            benchmark_groups = _select_benchmark_groups(
                benchmark_groups, args.only_categories
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.tasks_per_category is not None:
            benchmark_groups = {
                category_id: _stable_task_subset(tasks, args.tasks_per_category)
                for category_id, tasks in benchmark_groups.items()
            }

        # Order benchmarks according to the parent suite's task_groups if available
        ordered_benchmarks: list[str] = []
        if selected_suite and selected_suite.get("task_groups"):
            for tg in selected_suite["task_groups"]:
                tg_id = tg.get("id")
                if tg_id in benchmark_groups and tg_id not in ordered_benchmarks:
                    ordered_benchmarks.append(tg_id)
        for cat_id in benchmark_groups:
            if cat_id not in ordered_benchmarks:
                ordered_benchmarks.append(cat_id)

        print(f"✓ Suite '{selected_suite_id}' consists of {len(ordered_benchmarks)} individual benchmark(s):", flush=True)
        for idx, b_name in enumerate(ordered_benchmarks, 1):
            print(f"   [{idx}/{len(ordered_benchmarks)}] {b_name} ({len(benchmark_groups[b_name])} tasks)", flush=True)

        startup_metadata = {"runtime_configuration_source": "external_server"}
        if args.launch_vllm:
            from src.runtime_configuration import managed_vllm
            startup_metadata = resources.enter_context(managed_vllm(
                model, args.vllm_base_url, startup_options, args.vllm_startup_timeout,
            ))
        request_options = ({"response_format": args.response_format} if engine in {"ollama", "vllm"} else {})
        sampling = {"temperature": args.temperature if args.temperature is not None else 0.0,
                    "seed": args.seed if args.seed is not None else 42}

        # Prepare hardware info before benchmark execution
        if hardware_info is None and engine != "openai-compatible":
            try:
                hw_url = os.environ.get(
                    "VLLM_HARDWARE_URL" if engine == "vllm" else "OLLAMA_HARDWARE_URL"
                )
                base_engine_url = os.environ.get("OLLAMA_BASE_URL") if engine == "ollama" else os.environ.get("VLLM_BASE_URL")
                is_remote = False
                if base_engine_url:
                    parsed_u = urlsplit(base_engine_url)
                    if parsed_u.hostname and parsed_u.hostname not in ("127.0.0.1", "localhost", "0.0.0.0"):
                        is_remote = True
                        if not hw_url:
                            hw_url = f"{parsed_u.scheme or 'http'}://{parsed_u.hostname}:8081"

                if is_remote or hw_url:
                    if hw_url:
                        hardware_info = detect_hardware_from_endpoint(hw_url)
                else:
                    hardware_info = detect_hardware()
            except Exception as exc:
                print(f"Warning: hardware detection failed again ({exc}); submitting without hardware stats.")

        from src.quality_eval import score_runs
        from src.metrics_sampler import GpuMemorySampler, MetricsSampler

        vllm_metrics_url = os.environ.get("VLLM_METRICS_URL")
        vllm_hardware_url = (
            os.environ.get("VLLM_HARDWARE_URL" if engine == "vllm" else "OLLAMA_HARDWARE_URL")
            if engine != "openai-compatible" else None
        )
        base_engine_url = (
            os.environ.get("OLLAMA_BASE_URL") if engine == "ollama"
            else os.environ.get("VLLM_BASE_URL") if engine == "vllm"
            else None
        )
        if not vllm_hardware_url and base_engine_url:
            parsed_engine_url = urlsplit(base_engine_url)
            if parsed_engine_url.hostname and parsed_engine_url.hostname not in (
                "127.0.0.1", "localhost", "0.0.0.0"
            ):
                vllm_hardware_url = (
                    f"{parsed_engine_url.scheme or 'http'}://{parsed_engine_url.hostname}:8081"
                )
        
        if engine == "vllm" and not vllm_metrics_url:
            vllm_base = os.environ.get("VLLM_BASE_URL")
            if vllm_base:
                from urllib.parse import urlparse
                parsed = urlparse(vllm_base)
                if parsed.hostname:
                    vllm_root = vllm_base.rstrip("/")
                    if vllm_root.endswith("/v1"):
                        vllm_root = vllm_root[:-3]
                    vllm_metrics_url = f"{vllm_root}/metrics"
            if not vllm_metrics_url:
                vllm_metrics_url = "http://127.0.0.1:8000/metrics"

        context_keywords = (
            "maximum context length",
            "context_length_exceeded",
            "prompt is too long",
            "exceeds the context window",
            "400 client error: bad request",
        )
        crash_keywords = (
            "connection refused",
            "connection reset by peer",
            "503 service unavailable",
            "remoteprocesserror",
        )
        submitted_results = []
        outbox_results: list[tuple[str, Path]] = []
        for b_idx, bench_cat_id in enumerate(ordered_benchmarks, 1):
            bench_tasks = benchmark_groups[bench_cat_id]
            quality_tasks = {t["id"]: t for t in bench_tasks}
            needs_quality = True

            print(
                f"\n▶ Benchmark {b_idx}/{len(ordered_benchmarks)} · {bench_cat_id} · "
                f"{len(bench_tasks)} tasks",
                flush=True,
            )
            print(
                f"  Model {model} · {engine} · concurrency {concurrency} · "
                f"{mode} · {runs} repetition(s)",
                flush=True,
            )

            max_prompt_bytes = max(
                len((t.get("prompt") or t.get("user") or "").encode("utf-8"))
                for t in bench_tasks
            )
            needed_tokens = max_prompt_bytes + len((selected_prompt["system_prompt"] if selected_prompt else "").encode("utf-8")) + args.max_output_tokens + 128
            bench_min_ctx = args.context_length or max(needed_tokens, 4096)
            if args.context_length and args.context_length < needed_tokens:
                raise ValueError(f"Requested context {args.context_length} is below the conservative task/output budget {needed_tokens}; increase --context-length")

            sampler = (
                MetricsSampler(vllm_metrics_url, vllm_hardware_url=vllm_hardware_url)
                if engine == "vllm"
                else None
            )
            gpu_sampler = (
                GpuMemorySampler(hardware_url=vllm_hardware_url)
                if engine == "ollama"
                else None
            )
            if sampler:
                sampler.start()
            if gpu_sampler:
                gpu_sampler.start()
            try:
                if args.decision_mode:
                    benchmark_runs = []
                    for rep in range(runs):
                        batch = runner(
                            model=model,
                            prompt_category=bench_cat_id,
                            mode=mode,
                            num_runs=1,
                            concurrency=concurrency,
                            store_responses=needs_quality,
                            min_ctx=bench_min_ctx,
                            sampling={"temperature": 0.8, "seed": 100 + rep},
                            prompt_strategy=target_strategy,
                            custom_system_prompt=selected_prompt["system_prompt"] if selected_prompt else None,
                            max_output_tokens=args.max_output_tokens,
                            generation_timeout_seconds=args.generation_timeout_seconds,
                            custom_tasks=bench_tasks,
                            **openai_request_options,
                            **request_options,
                        )
                        for r in batch:
                            r["decision_repeat"] = rep
                        benchmark_runs.extend(batch)
                        if batch and all(
                            COMPLETION_BUDGET_EXHAUSTED in str(row.get("error") or "").lower()
                            for row in batch
                        ):
                            break
                else:
                    benchmark_runs = runner(
                        model=model,
                        prompt_category=bench_cat_id,
                        mode=mode,
                        num_runs=runs,
                        concurrency=concurrency,
                        store_responses=needs_quality,
                        min_ctx=bench_min_ctx,
                        sampling=sampling if engine in {"ollama", "vllm"} else None,
                        prompt_strategy=target_strategy,
                        custom_system_prompt=selected_prompt["system_prompt"] if selected_prompt else None,
                        max_output_tokens=args.max_output_tokens,
                        generation_timeout_seconds=args.generation_timeout_seconds,
                        custom_tasks=bench_tasks,
                        **openai_request_options,
                        **request_options,
                    )
            finally:
                if sampler:
                    sampler.stop()
                if gpu_sampler:
                    gpu_sampler.stop()

            deep_metrics = (
                sampler.aggregates() if sampler
                else gpu_sampler.aggregates() if gpu_sampler
                else {}
            )

            # Generate summary stats for this benchmark run
            provider_endpoint_hash = None
            if engine == "openai-compatible":
                from src.openai_compatible import normalize_base_url
                provider_endpoint_hash = hashlib.sha256(
                    normalize_base_url().encode("utf-8")
                ).hexdigest()
            submitted_engine_config = {
                "prompt_name": selected_prompt["name"] if selected_prompt else target_prompt_name,
                "prompt_strategy": selected_prompt.get("prompt_strategy") if selected_prompt else target_strategy,
                "max_output_tokens": str(args.max_output_tokens),
                "generation_timeout_seconds": str(args.generation_timeout_seconds),
                "tasks_per_category": (
                    str(args.tasks_per_category) if args.tasks_per_category is not None else "all"
                ),
                "task_selection_policy": "sha256_task_id" if args.tasks_per_category is not None else "complete",
                "temperature": "0.8" if args.decision_mode else str(sampling["temperature"]),
                "response_format": args.response_format,
                "seed_policy": (
                    f"100..{99 + runs}" if args.decision_mode
                    else "unset" if engine == "openai-compatible"
                    else str(sampling["seed"])
                ),
                "reasoning_policy": "native",
                "reasoning_effort": (
                    args.openai_reasoning_effort or "provider_default"
                    if engine == "openai-compatible" else None
                ),
                "completion_budget_policy": (
                    "provider_max_tokens" if engine == "openai-compatible"
                    else "single_configured_cap_clamped_to_context"
                ),
                "requested_min_context": (
                    None if engine == "openai-compatible" else str(bench_min_ctx)
                ),
                "context_length_policy": "explicit" if args.context_length is not None else "automatic_task_budget" if engine == "ollama" else "server_defined",
                "context_length_requested": str(args.context_length) if args.context_length is not None else None,
                "quantization": args.vllm_quantization or args.quantization,
                "model_revision": args.model_revision,
                "prompt_id": selected_prompt["id"] if selected_prompt else None,
                "prompt_sha256": selected_prompt["content_hash"] if selected_prompt else None,
            }
            submitted_engine_config.update(startup_metadata)
            if engine == "ollama":
                submitted_engine_config["runtime_configuration_source"] = "ollama_request_options"
                submitted_engine_config["context_source"] = "ollama_api_ps_or_unavailable"
            elif engine == "vllm":
                submitted_engine_config["context_source"] = "vllm_models_or_unavailable"
            if provider_endpoint_hash:
                submitted_engine_config["provider_endpoint_sha256"] = provider_endpoint_hash
            summary = summarise(
                model=model,
                prompt_category=bench_cat_id,
                mode=mode,
                runs=benchmark_runs,
                engine=engine,
                context_len=None,
                concurrency=concurrency,
                pipeline_config_id=args.pipeline_config_id,
                quantization=args.vllm_quantization or args.quantization,
                model_revision=args.model_revision,
                engine_config=submitted_engine_config,
            )
            if deep_metrics:
                summary.update(deep_metrics)

            # Quality scoring for this benchmark
            if needs_quality:
                if args.decision_mode:
                    import statistics
                    per_run_rates = []
                    for rep in range(runs):
                        batch = [r for r in benchmark_runs if r.get("decision_repeat") == rep]
                        per_run = score_runs(batch, quality_tasks)
                        if per_run["quality_success_rate"] is not None:
                            per_run_rates.append(per_run["quality_success_rate"])
                    decision_mode = {
                        "runs": runs,
                        "per_run_success_rate": per_run_rates,
                        "min_success_rate": min(per_run_rates),
                        "max_success_rate": max(per_run_rates),
                        "mean_success_rate": sum(per_run_rates) / len(per_run_rates),
                        "stdev_success_rate": (
                            statistics.stdev(per_run_rates)
                            if len(set(per_run_rates)) > 1
                            else 0.0
                        ),
                        "range": max(per_run_rates) - min(per_run_rates),
                    } if per_run_rates else None
                quality = score_runs(benchmark_runs, quality_tasks, capture_traces=args.capture_traces)
                summary.update(quality)
                if quality.get("scoring_error_count"):
                    error_ids = quality["scoring_error_task_ids"]
                    details = ", ".join(
                        f"{task_id} ({quality['task_scores'][task_id]['reason']})"
                        for task_id in error_ids[:5]
                    )
                    print(
                        f"\n❌ FATAL: {quality['scoring_error_count']} task result(s) could not be scored "
                        f"for benchmark '{bench_cat_id}': {details}. "
                        "Refusing submission; inspect the listed task/scorer contracts.",
                        file=sys.stderr,
                    )
                    continue
                if args.decision_mode:
                    summary["decision_mode"] = decision_mode
                    summary["num_runs"] = runs

            # Attach hardware facts
            if engine == "openai-compatible":
                summary["hardware"] = "Hosted API"
                summary["hw_detection_status"] = "not_applicable"
            elif hardware_info is not None:
                hw_dict = hardware_info.as_dict()
                for k, v in hw_dict.items():
                    if k == "platform_string":
                        summary["hw_platform"] = v
                    elif not k.startswith("hw_"):
                        summary[f"hw_{k}"] = v
                    else:
                        summary[k] = v
                summary["hw_detection_status"] = hardware_info.hw_detection_status
            else:
                summary["hw_detection_status"] = "failed"

            if args.experiment_tag:
                summary["experiment_tag"] = args.experiment_tag
            summary["run_type"] = args.run_type
            summary["concurrency_scope"] = args.concurrency_scope
            summary["approval_run_id"] = args.approval_run_id

            target_prompt_name = selected_prompt["name"] if selected_prompt else (args.prompt or "standard")
            target_strategy_name = selected_prompt.get("prompt_strategy") if selected_prompt else target_strategy
            summary["prompt_strategy"] = target_strategy_name
            summary["prompt_name"] = target_prompt_name
            if selected_prompt:
                summary.update({
                    "prompt_id": selected_prompt["id"],
                    "prompt_name": selected_prompt["name"],
                    "prompt_sha256": selected_prompt["content_hash"],
                    "system_prompt_snapshot": selected_prompt["system_prompt"],
                })
            summary["engine_config"] = submitted_engine_config


            # Crucial Suite & Category Metadata
            summary["prompt_category"] = bench_cat_id
            summary["category_id"] = bench_cat_id
            if selected_suite_id:
                summary["suite_id"] = selected_suite_id
                summary["scoreboard_suite_id"] = selected_suite_id
            if suite_name:
                summary["suite_name"] = suite_name

            # Error classification and guardrails for this benchmark
            errors = [r.get("error") for r in benchmark_runs if r.get("error")]
            total_reqs = len(benchmark_runs)
            error_cnt = len(errors)
            generation_timeout_count = sum(
                GENERATION_TIMEOUT_EXCEEDED in str(error).lower()
                for error in errors
            )
            generation_timeout_rate = (
                generation_timeout_count / total_reqs if total_reqs else 0.0
            )
            summary["generation_timeout_count"] = generation_timeout_count
            summary["generation_timeout_rate"] = generation_timeout_rate

            context_errors = [e for e in errors if any(kw in str(e).lower() for kw in context_keywords)]
            crash_errors = [e for e in errors if any(kw in str(e).lower() for kw in crash_keywords)]
            completion_budget_errors = [
                error for error in errors
                if COMPLETION_BUDGET_EXHAUSTED in str(error).lower()
            ]
            if context_errors:
                print(
                    f"\n❌ FATAL: {len(context_errors)}/{total_reqs} requests failed with Context Overflow (HTTP 400) on benchmark '{bench_cat_id}'. "
                    f"The context window is fundamentally undersized for this dataset. Aborting submission to prevent corrupted stats.",
                    file=sys.stderr
                )
                sys.exit(2)

            if crash_errors and len(crash_errors) >= 3:
                print(
                    f"\n❌ FATAL: {len(crash_errors)}/{total_reqs} requests failed with Engine Crash / Connection Refused (HTTP 503) on benchmark '{bench_cat_id}'. "
                    f"The inference engine has crashed or OOMed. Aborting submission.",
                    file=sys.stderr
                )
                sys.exit(3)

            if total_reqs > 0 and len(completion_budget_errors) == total_reqs:
                print(
                    f"\n❌ FATAL: the completion budget was exhausted before a complete answer "
                    f"for all {total_reqs} probe task(s) in '{bench_cat_id}'. Each task received one "
                    "generation at the shared configured output limit, clamped only by its available "
                    "context. Increase the common output limit and, when necessary, the engine context "
                    "window; native reasoning was not disabled. Remaining categories were not started.",
                    file=sys.stderr,
                )
                sys.exit(5)

            if generation_timeout_count:
                print(
                    f"\n⚠️ TIMEOUT EVIDENCE: {generation_timeout_count}/{total_reqs} attempted "
                    f"tasks timed out ({generation_timeout_rate:.1%}) on benchmark "
                    f"'{bench_cat_id}'. They remain failed outcomes and this partial category "
                    "will be submitted.",
                    file=sys.stderr,
                )

            if _submission_blocked_by_task_errors(errors, total_reqs):
                infrastructure_error_count = error_cnt - generation_timeout_count - len(completion_budget_errors)
                infrastructure_error_rate = (
                    infrastructure_error_count / total_reqs if total_reqs else 1.0
                )
                print(
                    f"\n⚠️ ABORT: Infrastructure execution failure rate too high on benchmark "
                    f"'{bench_cat_id}' ({infrastructure_error_count}/{total_reqs}, "
                    f"{infrastructure_error_rate:.1%}). Refusing submission because these failures "
                    "may indicate broken infrastructure or invalid results.",
                    file=sys.stderr
                )
                continue

            out_file = _save_to_outbox(summary)
            outbox_results.append((bench_cat_id, out_file))
            print(f"  ✓ Secured locally · {out_file.name}", flush=True)

            if not submission_contract_ready:
                try:
                    _check_submission_readiness(backend_url, api_key)
                    submission_contract_ready = True
                except RuntimeError as exc:
                    print(
                        f"  ↳ Retained for replay · ingestion not ready · {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

            print(f"  … Submitting {bench_cat_id}", flush=True)
            try:
                _, res_json = _post_outbox_file(
                    out_file,
                    backend_url,
                    api_key,
                    args.submission_timeout_seconds,
                )
                run_id = res_json.get("id") or res_json.get("status")
                q_disp = f"{summary.get('quality_score', 0)*100:.1f}%" if summary.get('quality_score') is not None else "N/A"
                tps_disp = f"{summary.get('median_decode_tps', 0):.1f} tok/s" if summary.get('median_decode_tps') is not None else "N/A"
                ttft_disp = f"{summary.get('median_ttft_s', 0):.3f}s" if summary.get('median_ttft_s') is not None else "N/A"
                print(
                    f"  ✓ Submitted {bench_cat_id} · run {run_id} · quality {q_disp} · "
                    f"decode {tps_disp} · TTFT {ttft_disp}",
                    flush=True,
                )
                submitted_results.append((bench_cat_id, res_json))

                if any(RESULTS_DIR.glob("fallback_*.json")):
                    print("  ↻ Backend recovered · draining saved results", flush=True)
                    replayed, _ = _replay_fallbacks(
                        backend_url, api_key, args.submission_timeout_seconds
                    )
                    for replayed_payload, replayed_ack in replayed:
                        replayed_category = replayed_payload.get("prompt_category")
                        if (
                            replayed_payload.get("experiment_tag") == args.experiment_tag
                            and replayed_payload.get("model") == model
                            and replayed_category in ordered_benchmarks
                        ):
                            submitted_results.append((replayed_category, replayed_ack))
            except requests.HTTPError as req_err:
                status = req_err.response.status_code if req_err.response is not None else None
                error_code, message, error_id = _response_error_details(req_err.response)
                reference = f" · error {error_id}" if error_id else ""
                if status is not None and status < 500:
                    print(
                        f"❌ Control plane rejected benchmark '{bench_cat_id}' · HTTP {status} · "
                        f"{error_code}{reference} · {message}",
                        file=sys.stderr,
                    )
                    if status not in {409, 429}:
                        raise SystemExit(1)
                else:
                    print(
                        f"  ✗ Submission failed · {bench_cat_id} · HTTP {status} · "
                        f"{error_code}{reference} · {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                    submission_contract_ready = False
                print(f"  ↳ Retained for replay · {out_file.name}", file=sys.stderr)
            except requests.exceptions.RequestException as req_err:
                submission_contract_ready = False
                print(
                    f"  ✗ Submission acknowledgement uncertain · {bench_cat_id} · "
                    f"{type(req_err).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                print(f"  ↳ Retained for idempotent replay · {out_file.name}", file=sys.stderr)

        deferred_results = [
            (category, path.name)
            for category, path in outbox_results
            if path.exists()
        ]
        print(
            f"\n✓ Suite execution completed · {len(submitted_results)} submitted · "
            f"{len(deferred_results)} saved for replay",
            flush=True,
        )
        missing_categories = _missing_category_rows(
            ordered_benchmarks, submitted_results, outbox_results
        )
        if missing_categories:
            print(
                "❌ Suite incomplete: no submitted or saved row for "
                f"{', '.join(missing_categories)}. Review the category error above; "
                "rerun the missing categories after the cause is fixed.",
                file=sys.stderr,
            )
            raise SystemExit(6)
        if deferred_results:
            raise SystemExit(4)

    except Exception as e:
        print(f"Error during benchmark or submission: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    try:
        with ExitStack() as resources:
            _main(resources)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed outbox files remain available for replay.", file=sys.stderr)
        raise SystemExit(130)
    except EOFError:
        print("Input ended. Supply all required CLI flags for a non-interactive run.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

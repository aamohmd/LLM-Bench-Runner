"""Local vLLM process lifecycle for explicitly managed benchmark runs."""
from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

import requests


@contextmanager
def managed_vllm(model: str, base_url: str, settings: dict, timeout: int = 600):
    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in {"", "/v1"} or not parsed.port):
        raise ValueError("Managed vLLM needs an explicit http://localhost:<unused-port>/v1 endpoint on the runner machine")
    # Never attach to or stop an existing inference service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", parsed.port))
    root = f"http://127.0.0.1:{parsed.port}"
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
               "--host", "127.0.0.1", "--port", str(parsed.port), "--generation-config", "vllm"]
    for key, value in settings.items():
        if value is None:
            continue
        if key == "prefix_caching":
            # vLLM versions expose different negative spellings; inspect this installation.
            help_result = subprocess.run(command[:3] + ["--help"], capture_output=True, text=True, timeout=60)
            if help_result.returncode:
                raise RuntimeError("Unable to inspect installed vLLM; install a compatible GPU-enabled vLLM environment")
            flag = "--enable-prefix-caching"
            if value == "disabled":
                flag = "--no-enable-prefix-caching" if "--no-enable-prefix-caching" in help_result.stdout else "--disable-prefix-caching"
            if flag not in help_result.stdout:
                raise ValueError(f"Installed vLLM does not support {flag}")
            command.append(flag)
        else:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    metadata = {"runtime_configuration_source": "runner_launched_vllm",
                "startup_settings_source": "process_arguments_not_memory_measurements",
                "startup_generation_config": "vllm"}
    metadata.update({f"startup_{key}": str(value) for key, value in settings.items() if value is not None})
    print("Starting a dedicated local vLLM process; it will stop when this benchmark exits.", flush=True)
    process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
    try:
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited before readiness (code {process.returncode}); inspect the startup output above")
            try:
                response = requests.get(root + "/v1/models", timeout=2)
                response.raise_for_status()
                payload = response.json()
                models = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(models, list) and any(isinstance(item, dict) and item.get("id") == model for item in models):
                    break
            except (requests.RequestException, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("vLLM startup timed out; inspect model compatibility and available GPU memory")
            time.sleep(1)
        yield metadata
    finally:
        # Only this owned process group is touched, including on Ctrl+C or failed submission.
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
            process.wait(timeout=20)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=10)

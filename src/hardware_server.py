"""Companion hardware-info HTTP server — runs on the GPU host.

Serves a single JSON endpoint so the benchmark backend can discover the
hardware facts of a remote vLLM instance without SSH or cloud API access.

Endpoints:
  GET /health   → {"status": "ok"}
  GET /hardware → HardwareInfo as JSON

Design decisions:
- detect_hardware() is called ONCE at startup, not per request.
- If detect_hardware() fails (e.g. nvidia-smi not ready yet at container
  start), the server catches the exception, logs it, and starts anyway with
  null GPU fields — it does NOT crash-loop. A crash-looping companion would
  appear as "remote_unreachable" to the backend, which is the safe outcome,
  but staying up with null GPU fields is more informative.
- Stdlib only (http.server) — no FastAPI dependency on the GPU host.
- Binds 0.0.0.0 so it's reachable from the Mac backend over the network.
  Restrict access via EC2 security groups (port 8081 to your IP only).

Usage (started by docker compose --profile gpu):
  python src/hardware_server.py [--port 8081]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_hardware_info() -> dict:
    """Detect hardware once at startup. Never raises — returns null fields on any error."""
    try:
        from src.hardware_detect import detect_hardware
        hw = detect_hardware(status="remote")
        logger.info("Hardware detected: cpu=%s gpu=%s aws=%s",
                    hw.cpu_model, hw.gpu_name, hw.aws_instance_type)
        return hw.as_dict()
    except Exception as exc:
        logger.error("Hardware detection failed at startup: %s", exc)
        logger.warning("Serving null GPU fields — hw_detection_status will be 'remote_startup_error'")
        import platform
        return {
            "platform_string": platform.platform(),
            "hw_detection_status": "remote_startup_error",
            "cpu_model": None,
            "cpu_physical_cores": None,
            "cpu_logical_cores": None,
            "ram_gb": None,
            "gpu_name": None,
            "gpu_memory_mb": None,
            "gpu_driver_version": None,
            "gpu_count": None,
            "gpu_compute_capability": None,
            "aws_instance_type": None,
        }

def _poll_gpu_mem() -> float | None:
    # Fast path: aggregate NVML memory across all visible devices.
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        used_bytes = sum(
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).used
            for index in range(device_count)
        )
        return round(float(used_bytes) / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    # Fallback path: subprocess nvidia-smi
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=2
        )
        values = [float(line.strip()) for line in out.decode().splitlines() if line.strip()]
        return sum(values) if values else None
    except Exception:
        pass

    # ROCm reports used VRAM in bytes for each card.
    try:
        import subprocess
        output = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.STDOUT,
            timeout=2,
        )
        data = json.loads(output.decode())
        used_bytes = 0
        found = False
        for name, values in data.items():
            if not name.startswith("card") or not isinstance(values, dict):
                continue
            for key in ("VRAM Total Used Memory (B)", "VRAM Used Memory (B)", "GPU Memory Used (B)"):
                value = values.get(key)
                if value is not None:
                    used_bytes += int(value)
                    found = True
                    break
        return round(used_bytes / (1024.0 * 1024.0), 2) if found else None
    except Exception:
        return None


_HARDWARE_PAYLOAD: dict = {}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/hardware":
            self._respond(200, _HARDWARE_PAYLOAD)
        elif self.path == "/metrics":
            mem = _poll_gpu_mem()
            self._respond(200, {"gpu_mem_mb": mem})
        elif self.path == "/vllm-metrics":
            import os
            from prometheus_client import CollectorRegistry, generate_latest
            from prometheus_client.multiprocess import MultiProcessCollector

            registry = CollectorRegistry()
            prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
            if prom_dir and os.path.exists(prom_dir):
                MultiProcessCollector(registry, path=prom_dir)
            
            encoded = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def main() -> None:
    from http.server import ThreadingHTTPServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    global _HARDWARE_PAYLOAD
    _HARDWARE_PAYLOAD = _load_hardware_info()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    logger.info("Hardware server listening on 0.0.0.0:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()

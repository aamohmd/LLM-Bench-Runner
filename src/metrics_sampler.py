import json
import time
import subprocess
import requests
import threading
from prometheus_client.parser import text_string_to_metric_families

class MetricsSampler:
    def __init__(self, vllm_metrics_url: str, vllm_hardware_url: str | None = None, interval_s: float = 1.0):
        self.vllm_metrics_url = vllm_metrics_url
        self.vllm_hardware_url = vllm_hardware_url
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._running = False
        self._thread = None
        self._previous_itl_totals: tuple[float, float] | None = None
        self._last_itl_delta_count: float | None = None
        self._last_itl_was_baseline = False
        from src.engines import create_http_session
        self._session = create_http_session(pool_connections=20, pool_maxsize=20)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        while self._running:
            gpu_mem = self._poll_gpu_mem()
            kv_cache, itl, kv_ok, itl_ok = self._poll_vllm_metrics()
            self.samples.append({
                "ts": time.time(),
                "gpu_mem_mb": gpu_mem,
                "kv_cache_pct": kv_cache,
                "itl_ms": itl,
                "itl_token_count": self._last_itl_delta_count,
                "itl_baseline": self._last_itl_was_baseline,
                "kv_ok": kv_ok,
                "itl_ok": itl_ok,
            })
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        try:
            self._session.close()
        except Exception:
            pass

    def _poll_gpu_mem(self) -> float | None:
        if self.vllm_hardware_url:
            try:
                resp = self._session.get(f"{self.vllm_hardware_url}/metrics", timeout=5.0)
                if resp.status_code == 200:
                    return resp.json().get("gpu_mem_mb")
            except Exception:
                pass
            return None

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.STDOUT
            )
            return float(out.decode().strip().splitlines()[0])
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            return None

    def _poll_vllm_metrics(self) -> tuple[float | None, float | None, bool, bool]:
        kv_cache_pct = None
        itl_ms = None
        self._last_itl_delta_count = None
        self._last_itl_was_baseline = False
        kv_ok = False
        itl_ok = False
        try:
            resp = self._session.get(self.vllm_metrics_url, timeout=2.0)
            if resp.status_code == 200:
                for family in text_string_to_metric_families(resp.text):
                    if family.name in ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"):
                        values = []
                        for sample in family.samples:
                            values.append(float(sample.value) * 100.0)
                        if values:
                            kv_cache_pct = max(values)
                            kv_ok = True
                    elif family.name == "vllm:time_per_output_token_seconds":
                        v_sum = 0.0
                        v_count = 0.0
                        for sample in family.samples:
                            if sample.name.endswith("_sum"):
                                v_sum += float(sample.value)
                            elif sample.name.endswith("_count"):
                                v_count += float(sample.value)
                        previous = self._previous_itl_totals
                        self._previous_itl_totals = (v_sum, v_count)
                        self._last_itl_was_baseline = previous is None
                        if previous is not None and v_count > previous[1] and v_sum >= previous[0]:
                            delta_count = v_count - previous[1]
                            itl_ms = ((v_sum - previous[0]) / delta_count) * 1000.0
                            self._last_itl_delta_count = delta_count
                            itl_ok = True
        except Exception:
            pass
        return kv_cache_pct, itl_ms, kv_ok, itl_ok

    def aggregates(self) -> dict:
        if not self.samples:
            return {}
            
        gpu = [s["gpu_mem_mb"] for s in self.samples if s["gpu_mem_mb"] is not None]
        kv = [s["kv_cache_pct"] for s in self.samples if s["kv_cache_pct"] is not None]
        itl = sorted([s["itl_ms"] for s in self.samples if s["itl_ms"] is not None])
        
        peak_gpu = max(gpu) if gpu else None
        avg_gpu = sum(gpu) / len(gpu) if gpu else None
        peak_kv = max(kv) if kv else None
        avg_kv = sum(kv) / len(kv) if kv else None
        weighted_itl = [
            (s["itl_ms"], s.get("itl_token_count"))
            for s in self.samples
            if s.get("itl_ms") is not None and s.get("itl_token_count")
        ]
        avg_itl = (
            sum(value * count for value, count in weighted_itl) / sum(count for _, count in weighted_itl)
            if weighted_itl else (sum(itl) / len(itl) if itl else None)
        )
            
        gpu_failed = len(self.samples) - len(gpu)
        kv_failed = sum(not s.get("kv_ok", s.get("kv_cache_pct") is not None) for s in self.samples)
        itl_failed = sum(
            not s.get("itl_ok", s.get("itl_ms") is not None)
            for s in self.samples
            if not s.get("itl_baseline")
        )

        res = {
            "sample_count": len(self.samples),
            "gpu_failed_count": gpu_failed,
            "kv_failed_count": kv_failed,
            "itl_failed_count": itl_failed,
        }
        if peak_gpu is not None: res["peak_gpu_mem_mb"] = peak_gpu
        if avg_gpu is not None: res["avg_gpu_mem_mb"] = avg_gpu
        if peak_kv is not None: res["peak_kv_cache_pct"] = peak_kv
        if avg_kv is not None: res["avg_kv_cache_pct"] = avg_kv
        if avg_itl is not None: res["avg_itl_ms"] = avg_itl
        return res


class GpuMemorySampler:
    """Sample aggregate GPU memory while a category runs without vLLM metrics."""

    def __init__(self, hardware_url: str | None = None, interval_s: float = 1.0):
        self.hardware_url = hardware_url.rstrip("/") if hardware_url else None
        self.interval_s = interval_s
        self.samples: list[float] = []
        self.sample_count = 0
        self.gpu_failed_count = 0
        self._running = False
        self._thread: threading.Thread | None = None
        from src.engines import create_http_session
        self._session = create_http_session(pool_connections=4, pool_maxsize=4)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while self._running:
            self.sample_count += 1
            value = self._poll_gpu_mem()
            if value is not None:
                self.samples.append(value)
            else:
                self.gpu_failed_count += 1
            time.sleep(self.interval_s)

    def _poll_gpu_mem(self) -> float | None:
        if self.hardware_url:
            try:
                response = self._session.get(f"{self.hardware_url}/metrics", timeout=5.0)
                response.raise_for_status()
                value = response.json().get("gpu_mem_mb")
                return float(value) if value is not None else None
            except (requests.RequestException, TypeError, ValueError):
                return None
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            values = [float(line.strip()) for line in output.decode().splitlines() if line.strip()]
            return sum(values) if values else None
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            pass
        try:
            output = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            devices = json.loads(output.decode())
            used_bytes = 0
            found = False
            for name, values in devices.items():
                if not name.startswith("card") or not isinstance(values, dict):
                    continue
                for key in ("VRAM Total Used Memory (B)", "VRAM Used Memory (B)", "GPU Memory Used (B)"):
                    value = values.get(key)
                    if value is not None:
                        used_bytes += int(value)
                        found = True
                        break
            return used_bytes / (1024.0 * 1024.0) if found else None
        except (subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError, FileNotFoundError):
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join()
        self._session.close()

    def aggregates(self) -> dict:
        result = {
            "sample_count": self.sample_count,
            "gpu_failed_count": self.gpu_failed_count,
        }
        if self.samples:
            result["peak_gpu_mem_mb"] = max(self.samples)
            result["avg_gpu_mem_mb"] = sum(self.samples) / len(self.samples)
        return result

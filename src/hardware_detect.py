"""Hardware auto-detection for benchmark runs.

Returns a HardwareInfo dataclass. All hw_* fields are nullable — detection
is best-effort and gracefully returns None when a tool isn't available,
rather than raising or fabricating a value.

The hw_detection_status field is always populated and tells you exactly how
hardware was or wasn't detected for this run:
  "local"              — detected from the backend host (ollama)
  "local_assumed"      — vLLM job, no VLLM_HARDWARE_URL set, backend host used
  "remote"             — detected from companion endpoint (vLLM on remote host)
  "remote_unreachable" — companion configured but unreachable; all hw_* are null

GPU detection priority (first match wins):
  1. NVIDIA  — nvidia-smi; aggregates VRAM across all GPUs in the system.
  2. AMD     — rocm-smi; aggregates VRAM across all GPUs in the system.
  3. Apple Silicon — system_profiler + sysctl; unified memory treated as VRAM.

Cloud detection runs all three providers in parallel (all non-blocking, 200ms
timeout each) and records the first that responds, prefixed with the provider
name (e.g. "aws:p3.2xlarge", "gcp:n1-standard-4", "azure:Standard_NC6").
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    platform_string: str                  # platform.platform() — always populated
    hw_detection_status: str              # see module docstring, always populated
    cpu_model: str | None = None
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    ram_gb: float | None = None
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    gpu_driver_version: str | None = None
    gpu_count: int | None = None
    gpu_compute_capability: float | None = None
    aws_instance_type: str | None = None  # populated for any cloud provider (aws/gcp/azure prefix)

    @classmethod
    def remote_unreachable(cls) -> "HardwareInfo":
        """Factory: companion was configured but did not respond.

        All hw_* fields are null — this is honest "we don't know" data, not a
        placeholder. platform_string is still populated from the local host
        because it's free and helps with debugging which backend ran the job.
        """
        return cls(
            platform_string=platform.platform(),
            hw_detection_status="remote_unreachable",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform_string": self.platform_string,
            "hw_detection_status": self.hw_detection_status,
            "cpu_model": self.cpu_model,
            "cpu_physical_cores": self.cpu_physical_cores,
            "cpu_logical_cores": self.cpu_logical_cores,
            "ram_gb": self.ram_gb,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_driver_version": self.gpu_driver_version,
            "gpu_count": self.gpu_count,
            "gpu_compute_capability": self.gpu_compute_capability,
            "aws_instance_type": self.aws_instance_type,
        }


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def _cpu_model() -> str | None:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            brand = result.stdout.strip()
            if brand:
                return brand
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception as exc:
        logger.debug("CPU model detection failed: %s", exc)
    return platform.processor() or None


def _cpu_cores() -> tuple[int | None, int | None]:
    """Returns (physical_cores, logical_cores)."""
    logical = os.cpu_count()
    physical: int | None = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        # psutil not available — try subprocess on Linux
        if sys.platform.startswith("linux"):
            try:
                result = subprocess.run(
                    ["grep", "-c", "^processor", "/proc/cpuinfo"],
                    capture_output=True, text=True, timeout=3,
                )
                logical = int(result.stdout.strip()) or logical
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Physical core count failed: %s", exc)
    return physical, logical


def _ram_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("RAM detection via psutil failed: %s", exc)
    # Fallback for Linux without psutil
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
    except Exception as exc:
        logger.debug("RAM detection via /proc/meminfo failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# NVIDIA GPU detection
# ---------------------------------------------------------------------------

def _nvidia_gpu_info() -> tuple[str | None, int | None, str | None, int | None]:
    """Returns (gpu_name, total_gpu_memory_mb, gpu_driver_version, gpu_count).

    Aggregates VRAM across ALL GPUs in the system (summed, not just the first).
    gpu_name reflects the first/primary GPU; gpu_count is the total card count.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None, None, None, None

        rows = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        if not rows:
            return None, None, None, None

        gpu_count = len(rows)
        total_vram_mb = 0
        primary_name: str | None = None
        primary_driver: str | None = None

        for i, row in enumerate(rows):
            parts = [p.strip() for p in row.split(",")]
            if i == 0:
                primary_name = parts[0] if len(parts) > 0 else None
                primary_driver = parts[2] if len(parts) > 2 else None
            try:
                total_vram_mb += int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                pass

        # Annotate name when multiple GPUs are present so callers know the
        # gpu_memory_mb figure is an aggregate, not a single-card reading.
        display_name = primary_name
        if gpu_count > 1 and primary_name:
            display_name = f"{gpu_count}x {primary_name}"

        return display_name, total_vram_mb or None, primary_driver, gpu_count

    except FileNotFoundError:
        # nvidia-smi not present — CPU-only, Apple Silicon, or AMD
        return None, None, None, None
    except Exception as exc:
        logger.debug("nvidia-smi detection failed: %s", exc)
        return None, None, None, None


def _nvidia_compute_capability() -> float | None:
    """Return the first NVIDIA GPU compute capability, when available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return float(result.stdout.strip().splitlines()[0].strip())
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return None
    except Exception as exc:
        logger.debug("GPU compute capability detection failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# AMD ROCm GPU detection
# ---------------------------------------------------------------------------

def _amd_gpu_info() -> tuple[str | None, int | None, str | None, int | None]:
    """Returns (gpu_name, total_gpu_memory_mb, driver_version, gpu_count).

    Uses rocm-smi --showproductname and --showmeminfo vram.
    Aggregates VRAM across all detected AMD GPUs.
    """
    try:
        # rocm-smi JSON is the cleanest path — available in ROCm 5+
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            # Output is a dict keyed by card id: {"card0": {...}, "card1": {...}, ...}
            cards = {k: v for k, v in data.items() if k.startswith("card")}
            if cards:
                total_vram_mb = 0
                primary_name: str | None = None
                for i, (card_id, card_data) in enumerate(sorted(cards.items())):
                    if i == 0:
                        primary_name = (
                            card_data.get("Card series")
                            or card_data.get("Card model")
                            or card_data.get("GPU ID")
                        )
                    # VRAM is reported in bytes under several possible keys
                    for key in ("VRAM Total Memory (B)", "Total VRAM memory (B)", "GPU Memory Total (B)"):
                        vram_bytes = card_data.get(key)
                        if vram_bytes is not None:
                            try:
                                total_vram_mb += int(vram_bytes) // (1024 * 1024)
                            except (ValueError, TypeError):
                                pass
                            break
                gpu_count = len(cards)
                display_name = primary_name
                if gpu_count > 1 and primary_name:
                    display_name = f"{gpu_count}x {primary_name}"
                return display_name, total_vram_mb or None, None, gpu_count
    except FileNotFoundError:
        pass  # rocm-smi not installed
    except (json.JSONDecodeError, KeyError, Exception) as exc:
        logger.debug("rocm-smi JSON parse failed, trying plain text: %s", exc)

    # Fallback: rocm-smi plain text (older ROCm versions)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            total_vram_mb = 0
            gpu_count = 0
            for line in result.stdout.splitlines():
                line_lower = line.lower()
                if "total" in line_lower and "vram" in line_lower:
                    # e.g. "GPU[0]          : VRAM Total Memory (B): 17163091968"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            total_vram_mb += int(parts[-1].strip()) // (1024 * 1024)
                            gpu_count += 1
                        except ValueError:
                            pass
            if gpu_count > 0:
                return "AMD GPU", total_vram_mb or None, None, gpu_count
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("rocm-smi plain-text fallback failed: %s", exc)

    return None, None, None, None


# ---------------------------------------------------------------------------
# Apple Silicon GPU detection (unified memory architecture)
# ---------------------------------------------------------------------------

def _apple_silicon_gpu_info(ram_gb: float | None) -> tuple[str | None, int | None]:
    """Returns (gpu_name, gpu_memory_mb) for Apple Silicon, else (None, None).

    Apple Silicon uses unified memory shared between CPU and GPU. The effective
    GPU memory budget is the total system RAM, so gpu_memory_mb is set to that.
    gpu_name is read from system_profiler (the real chip name, e.g. "Apple M3 Pro").
    """
    if sys.platform != "darwin":
        return None, None

    # Only Apple Silicon (arm64); Intel Macs have no unified-memory GPU to report.
    if platform.machine() != "arm64":
        return None, None

    gpu_name: str | None = None
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            for entry in displays:
                # Try common keys — field name varies slightly across macOS versions
                name = (
                    entry.get("sppci_model")
                    or entry.get("_name")
                    or entry.get("spdisplays_vendor")
                )
                if name:
                    gpu_name = name
                    break
    except Exception as exc:
        logger.debug("system_profiler GPU detection failed: %s", exc)

    # Fallback name if system_profiler gave nothing
    if not gpu_name:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            brand = result.stdout.strip()
            # brand_string is e.g. "Apple M3 Pro" — good enough as a GPU label
            if brand and "Apple" in brand:
                gpu_name = brand
        except Exception as exc:
            logger.debug("sysctl brand_string fallback failed: %s", exc)

    if gpu_name is None:
        return None, None

    # Unified memory: total RAM is the GPU memory budget.
    gpu_memory_mb = int(ram_gb * 1024) if ram_gb else None
    return gpu_name, gpu_memory_mb


# ---------------------------------------------------------------------------
# Cloud instance-type detection  (AWS / GCP / Azure — 200ms timeout each)
# ---------------------------------------------------------------------------

def _detect_aws() -> str | None:
    """EC2 instance type via IMDS (IMDSv2 first, IMDSv1 fallback)."""
    base = "http://169.254.169.254"
    timeout = 0.2
    token: str | None = None
    try:
        put_req = Request(
            f"{base}/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urlopen(put_req, timeout=timeout) as resp:
            if resp.status == 200:
                token = resp.read().decode().strip()
    except Exception:
        pass
    try:
        headers = {}
        if token:
            headers["X-aws-ec2-metadata-token"] = token
        get_req = Request(f"{base}/latest/meta-data/instance-type", headers=headers)
        with urlopen(get_req, timeout=timeout) as resp:
            if resp.status == 200:
                return "aws:" + resp.read().decode().strip()
    except Exception:
        pass
    return None


def _detect_gcp() -> str | None:
    """GCP instance type via the Compute Engine metadata server."""
    try:
        req = Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
            headers={"Metadata-Flavor": "Google"},
        )
        with urlopen(req, timeout=0.2) as resp:
            if resp.status == 200:
                # Value is the full resource path; extract just the type suffix.
                value = resp.read().decode().strip()
                machine_type = value.split("/")[-1]
                return "gcp:" + machine_type
    except Exception:
        pass
    return None


def _detect_azure() -> str | None:
    """Azure VM size via the Instance Metadata Service."""
    try:
        req = Request(
            "http://169.254.169.254/metadata/instance/compute/vmSize"
            "?api-version=2021-02-01",
            headers={"Metadata": "true"},
        )
        with urlopen(req, timeout=0.2) as resp:
            if resp.status == 200:
                return "azure:" + resp.read().decode().strip()
    except Exception:
        pass
    return None


def _cloud_instance_type() -> str | None:
    """Probe AWS, GCP, and Azure concurrently; return the first that responds.

    All three probes run in parallel threads so the total latency is bounded
    by the slowest *responding* provider (~200ms), not the sum of all three.
    On a non-cloud machine all three time out and None is returned.
    """
    probes = {"aws": _detect_aws, "gcp": _detect_gcp, "azure": _detect_azure}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in probes.items()}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    # Cancel remaining — we already have what we need.
                    for f in futures:
                        f.cancel()
                    return result
            except Exception as exc:
                logger.debug("Cloud probe failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# GPU detection orchestration — NVIDIA → AMD → Apple Silicon (first match)
# ---------------------------------------------------------------------------

def _detect_gpu(
    ram_gb: float | None,
) -> tuple[str | None, int | None, str | None, int | None, float | None]:
    """Returns (gpu_name, gpu_memory_mb, gpu_driver, gpu_count, compute_capability).

    Tries NVIDIA first (most common for ML workloads), then AMD ROCm, then
    Apple Silicon. The first provider that returns a non-None gpu_name wins.
    """
    # 1. NVIDIA
    name, vram_mb, driver, count = _nvidia_gpu_info()
    if name:
        compute_cap = _nvidia_compute_capability()
        return name, vram_mb, driver, count, compute_cap

    # 2. AMD ROCm
    name, vram_mb, driver, count = _amd_gpu_info()
    if name:
        return name, vram_mb, driver, count, None

    # 3. Apple Silicon (unified memory)
    name, vram_mb = _apple_silicon_gpu_info(ram_gb)
    if name:
        return name, vram_mb, None, 1, None

    return None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hardware(status: str = "local") -> HardwareInfo:
    """Detect hardware facts from the local host.

    status: the hw_detection_status value to record. Pass "local_assumed"
    when calling for a vLLM job that has no VLLM_HARDWARE_URL configured.
    """
    physical_cores, logical_cores = _cpu_cores()
    ram_gb = _ram_gb()
    gpu_name, gpu_memory_mb, gpu_driver, gpu_count, gpu_compute_cap = _detect_gpu(ram_gb)
    return HardwareInfo(
        platform_string=platform.platform(),
        hw_detection_status=status,
        cpu_model=_cpu_model(),
        cpu_physical_cores=physical_cores,
        cpu_logical_cores=logical_cores,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        gpu_driver_version=gpu_driver,
        gpu_count=gpu_count,
        gpu_compute_capability=gpu_compute_cap,
        aws_instance_type=_cloud_instance_type(),
    )


def detect_hardware_from_endpoint(url: str, timeout: float = 5.0) -> HardwareInfo | None:
    """Fetch HardwareInfo from a companion hardware server.

    Returns None on any failure — the caller must NOT substitute local
    detection when this returns None for a remote-vLLM job.
    """
    try:
        req = Request(url.rstrip("/") + "/hardware")
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning("Hardware endpoint %s returned %s", url, resp.status)
                return None
            data = json.loads(resp.read())
        return HardwareInfo(**{k: data.get(k) for k in HardwareInfo.__dataclass_fields__})
    except Exception as exc:
        logger.warning("Hardware endpoint %s unreachable: %s", url, exc)
        return None

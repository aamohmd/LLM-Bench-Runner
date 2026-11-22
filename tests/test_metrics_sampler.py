import subprocess

from src.metrics_sampler import MetricsSampler


def test_vllm_sampler_sums_gpu_memory_and_bounds_subprocess(monkeypatch):
    def fake_output(command, **kwargs):
        assert kwargs["timeout"] == 5
        return b"100\n200\n"

    monkeypatch.setattr(subprocess, "check_output", fake_output)
    sampler = MetricsSampler("http://synthetic.invalid/metrics")
    try:
        assert sampler._poll_gpu_mem() == 300
    finally:
        sampler.stop()


def test_vllm_sampler_gpu_timeout_is_unavailable(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(subprocess, "check_output", timeout)
    sampler = MetricsSampler("http://synthetic.invalid/metrics")
    try:
        assert sampler._poll_gpu_mem() is None
    finally:
        sampler.stop()


def test_kv_cache_metric_is_reported_in_percentage_points(monkeypatch):
    class Response:
        status_code = 200
        text = '# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc 0.005\n'

    sampler = MetricsSampler("http://synthetic.invalid/metrics")
    monkeypatch.setattr(sampler._session, "get", lambda *args, **kwargs: Response())
    try:
        kv, _, ok, _ = sampler._poll_vllm_metrics()
        assert ok
        assert kv == 0.5
    finally:
        sampler.stop()

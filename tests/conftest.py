"""Offline fixtures: unit tests cannot contact an inference service or platform."""
import pytest
import requests


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch, tmp_path):
    from scripts import run_remote_benchmark as cli
    from src import benchmark

    def reject_network(*args, **kwargs):
        raise AssertionError("Live network access is disabled in unit tests; mock the endpoint")

    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
    def unavailable_metadata(*args, **kwargs):
        raise requests.ConnectionError("Metadata endpoint is not supplied by this test")

    monkeypatch.setattr(benchmark.CLIENT, "get", unavailable_metadata)
    monkeypatch.setattr(benchmark.CLIENT, "post", unavailable_metadata)
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "outbox")
    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "measurements")
    for key in ("API_KEY", "BACKEND_URL", "BENCHMARK_API_URL", "OPENAI_API_KEY", "OLLAMA_NUM_PARALLEL"):
        monkeypatch.delenv(key, raising=False)

"""Failure cases that must not lose evidence, start work, or leak process ownership."""
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests

from scripts import run_remote_benchmark as cli
from src import engines, runtime_configuration as lifecycle


def response(status, payload=None, text=None):
    result = requests.Response()
    result.status_code = status
    result._content = text.encode() if text is not None else json.dumps(payload).encode()
    return result


@pytest.mark.parametrize("status,body,text", [
    (200, None, "<html>Sign in</html>"), (204, None, ""),
    (200, [], None), (200, None, None), (200, {}, None),
    (200, {"status": "created"}, None),
    (200, {"status": "created", "id": ""}, None),
    (200, {"status": "created", "id": 17}, None),
    (200, {"status": "accepted", "id": "run-id"}, None),
    (302, {"status": "created", "id": "run-id"}, None),
    (401, {"detail": "Expired key"}, None),
    (409, {"detail": "Identity conflict"}, None),
    (429, {"detail": "Rate limited"}, None),
    (503, {"detail": "Unavailable"}, None),
])
def test_outbox_survives_invalid_or_rejected_ack(monkeypatch, status, body, text):
    payload = {"model": "fixture-model", "experiment_tag": "synthetic-ack-check"}
    path = cli._save_to_outbox(payload)
    original = path.read_bytes()
    post = Mock(return_value=response(status, body, text))
    monkeypatch.setattr(cli.requests, "post", post)
    with pytest.raises(requests.RequestException):
        cli._post_outbox_file(path, "https://control.example", "fixture-key", 5)
    assert path.read_bytes() == original
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("status", ["created", "already_created"])
def test_acknowledged_result_removes_only_its_outbox_file(monkeypatch, status):
    path = cli._save_to_outbox({"model": "fixture-model"})
    other = cli._save_to_outbox({"model": "another-fixture"})
    monkeypatch.setattr(cli.requests, "post", lambda *a, **k: response(200, {"status": status, "id": "saved-id"}))
    payload, ack = cli._post_outbox_file(path, "https://control.example", "fixture-key", 5)
    assert ack["id"] == "saved-id" and payload["submission_id"]
    assert not path.exists() and other.exists()


def test_already_removed_acknowledged_file_is_not_reported_as_failure(monkeypatch):
    path = cli._save_to_outbox({"model": "fixture-model"})

    def acknowledge(*args, **kwargs):
        path.unlink()  # Another replay received the same idempotent acknowledgement.
        return response(200, {"status": "already_created", "id": "saved-id"})

    monkeypatch.setattr(cli.requests, "post", acknowledge)
    assert cli._post_outbox_file(path, "https://control.example", "fixture-key", 5)[1]["id"] == "saved-id"


def test_preflight_does_not_forward_credentials_through_redirect(monkeypatch):
    get = Mock(return_value=response(302, text="<html>Moved</html>"))
    monkeypatch.setattr(cli.requests, "get", get)
    with pytest.raises(RuntimeError):
        cli._check_submission_readiness("https://control.example", "fixture-key")
    assert get.call_args.kwargs["allow_redirects"] is False


def test_malformed_scorer_catalog_stops_before_inference(monkeypatch):
    get = Mock(return_value=response(200, []))
    monkeypatch.setattr(cli.requests, "get", get)
    monkeypatch.setattr(sys, "argv", ["runner", "--backend-url", "https://control.example", "--api-key", "fixture-key", "--experiment-tag", "synthetic-invalid-catalog"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert get.call_count == 1


def test_replay_continues_past_corrupt_file_without_removing_it(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path)
    corrupt = tmp_path / "fallback_0.json"
    corrupt.write_text("{truncated")
    valid = cli._save_to_outbox({"model": "fixture-model"})
    monkeypatch.setattr(cli.requests, "post", lambda *a, **k: response(200, {"status": "created", "id": "saved-id"}))
    successes, failures = cli._replay_fallbacks("https://control.example", "fixture-key", 5)
    assert len(successes) == 1 and failures == 1
    assert corrupt.exists() and not valid.exists()


def test_atomic_write_failure_preserves_previous_result(monkeypatch, tmp_path):
    path = tmp_path / "outbox.json"
    cli._atomic_write_json(path, {"version": 1})
    monkeypatch.setattr(cli.os, "replace", Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(OSError):
        cli._atomic_write_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_outbox_payload_is_private_with_permissive_umask():
    previous = os.umask(0)
    try:
        path = cli._save_to_outbox({"system_prompt_snapshot": "synthetic private prompt"})
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("url", ["not a URL", "ftp://control.example", "http://user:key@control.example", "https://control.example?key=x", "https://control.example/#page", "http://control.example:invalid"])
def test_bad_platform_address_rejected_before_http(monkeypatch, url):
    get = Mock(side_effect=AssertionError("No request may start"))
    monkeypatch.setattr(cli.requests, "get", get)
    monkeypatch.setattr(sys, "argv", ["runner", "--backend-url", url])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    get.assert_not_called()


def test_missing_input_never_attempts_terminal_prompt(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit) as caught:
        cli._check_interactive()
    assert caught.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "many", "1.5"])
def test_invalid_parallel_environment_stops_before_http(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", value)
    get = Mock(side_effect=AssertionError("No network may start"))
    monkeypatch.setattr(cli.requests, "get", get)
    monkeypatch.setattr(sys, "argv", ["runner", "--engine", "ollama"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    get.assert_not_called()


def test_prompt_cancellation_is_distinct_from_default_input():
    with pytest.raises(SystemExit) as caught:
        cli._ask(SimpleNamespace(ask=lambda: None))
    assert caught.value.code == 130
    assert cli._ask(SimpleNamespace(ask=lambda: "1")) == "1"


def test_ctrl_c_cleans_owned_resources_and_exits_130(monkeypatch, capsys):
    cleaned = []

    @contextmanager
    def resource():
        try:
            yield
        finally:
            cleaned.append(True)

    def interrupt(stack):
        stack.enter_context(resource())
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_main", interrupt)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 130 and cleaned == [True]
    assert "outbox files remain" in capsys.readouterr().err


@pytest.mark.parametrize("document", [None, [], {"data": None}, {"data": [None, "bad item"]}, {"data": [{"id": "other"}]}])
def test_malformed_vllm_readiness_waits_then_cleans_up(monkeypatch, document):
    monkeypatch.setattr(lifecycle.socket, "socket", MagicMock())
    process = Mock(pid=99123)
    process.poll.return_value = None
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **k: process)
    kill = Mock()
    monkeypatch.setattr(lifecycle.os, "killpg", kill)
    monkeypatch.setattr(lifecycle.requests, "get", lambda *a, **k: response(200, document))
    ticks = iter([0, 2])
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="startup timed out"):
        with lifecycle.managed_vllm("fixture-model", "http://localhost:19011/v1", {}, timeout=1):
            pytest.fail("Benchmark must not start")
    kill.assert_called_once_with(process.pid, lifecycle.signal.SIGTERM)


def test_cleanup_exit_race_does_not_mask_benchmark_error(monkeypatch):
    monkeypatch.setattr(lifecycle.socket, "socket", MagicMock())
    process = Mock(pid=99123)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("vllm", 20), None]
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **k: process)
    kill = Mock(side_effect=[None, ProcessLookupError()])
    monkeypatch.setattr(lifecycle.os, "killpg", kill)
    monkeypatch.setattr(lifecycle.requests, "get", lambda *a, **k: response(200, {"data": [{"id": "fixture-model"}]}))
    with pytest.raises(RuntimeError, match="benchmark failure"):
        with lifecycle.managed_vllm("fixture-model", "http://localhost:19011/v1", {}):
            raise RuntimeError("benchmark failure")
    assert process.wait.call_count == 2


def test_retry_releases_failed_http_connections(monkeypatch):
    first, second = response(503, {}), response(200, {})
    first.close = Mock()
    client = Mock()
    client.post.side_effect = [first, second]
    monkeypatch.setattr(engines.time, "sleep", lambda *a: None)
    assert engines.post_with_retry(client, "https://engine.example", retries=1) is second
    first.close.assert_called_once()

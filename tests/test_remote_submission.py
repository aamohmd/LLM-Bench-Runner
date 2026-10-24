from __future__ import annotations

import json

import pytest
import requests

from scripts import run_remote_benchmark as runner


def _response(status: int, body: object | None = None, *, text: str = "", headers: dict | None = None):
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    response._content = (
        json.dumps(body).encode("utf-8") if body is not None else text.encode("utf-8")
    )
    response.headers.setdefault("Content-Type", "application/json" if body is not None else "text/plain")
    return response


def test_submission_readiness_accepts_current_contract(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runner.requests,
        "get",
        lambda *args, **kwargs: _response(
            200,
            {
                "status": "ready",
                "contract_version": runner.INGESTION_CONTRACT_VERSION,
                "persistence": "reachable",
                "git_revision": "abc123",
            },
        ),
    )

    readiness = runner._check_submission_readiness("https://control.example", "secret")

    assert readiness["git_revision"] == "abc123"
    assert (
        f"Result ingestion ready · contract {runner.INGESTION_CONTRACT_VERSION} · abc123"
        in capsys.readouterr().out
    )


def test_submission_readiness_rejects_old_backend_before_inference(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.requests,
        "get",
        lambda *args, **kwargs: _response(404, {"detail": "Not Found"}),
    )

    with pytest.raises(RuntimeError, match="older ingestion contract"):
        runner._check_submission_readiness("https://control.example", "secret")


def test_submission_readiness_surfaces_structured_backend_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.requests,
        "get",
        lambda *args, **kwargs: _response(
            503,
            {
                "detail": "Benchmark result storage is not reachable.",
                "error_code": "persistence_unavailable",
                "error_id": "error-123",
            },
        ),
    )

    with pytest.raises(RuntimeError) as raised:
        runner._check_submission_readiness("https://control.example", "secret")

    assert "persistence_unavailable" in str(raised.value)
    assert "error-123" in str(raised.value)
    assert "No benchmark tasks were started" in str(raised.value)


def test_submission_readiness_classifies_network_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise requests.exceptions.SSLError("private TLS diagnostics")

    monkeypatch.setattr(runner.requests, "get", fail)

    with pytest.raises(RuntimeError, match=r"Cannot verify result ingestion \(SSLError\)"):
        runner._check_submission_readiness("https://control.example", "secret")


@pytest.mark.parametrize(
    "body",
    [
        ["not", "an", "object"],
        {"status": "ready", "contract_version": 1},
        {"status": "maintenance", "contract_version": runner.INGESTION_CONTRACT_VERSION},
    ],
)
def test_submission_readiness_rejects_invalid_contract_documents(monkeypatch, body) -> None:
    monkeypatch.setattr(
        runner.requests,
        "get",
        lambda *args, **kwargs: _response(200, body),
    )

    with pytest.raises(RuntimeError, match="[Nn]o benchmark tasks were started"):
        runner._check_submission_readiness("https://control.example", "secret")


def test_response_error_details_ignores_html_and_uses_error_header() -> None:
    response = _response(
        500,
        text="<html>proxy failure</html>",
        headers={"X-Error-ID": "proxy-456"},
    )

    code, message, error_id = runner._response_error_details(response)

    assert code == "http_500"
    assert message == "No diagnostic detail was returned."
    assert error_id == "proxy-456"


def test_isolated_task_failure_does_not_block_completed_category_submission() -> None:
    assert runner._submission_error_rate_exceeded(1, 150) is False
    assert runner._submission_error_rate_exceeded(8, 150) is True
    assert runner._submission_error_rate_exceeded(0, 0) is True


def test_generation_timeouts_are_submittable_failed_outcomes() -> None:
    timeouts = [
        "generation timeout exceeded before a complete assistant answer (120s limit)"
    ] * 10

    assert runner._submission_blocked_by_task_errors(timeouts, 110) is False
    assert runner._submission_blocked_by_task_errors(
        [*timeouts, *("unexpected protocol failure" for _ in range(6))],
        110,
    ) is True


def test_completion_budget_failures_are_submittable_failed_outcomes() -> None:
    budget_failures = [
        "completion budget exhausted before a complete assistant answer "
        "(during assistant content; requested=256, available=7109)"
    ] * 16

    assert runner._submission_blocked_by_task_errors(budget_failures, 150) is False
    assert runner._submission_blocked_by_task_errors(
        [*budget_failures, *("unexpected protocol failure" for _ in range(8))],
        150,
    ) is True


def test_missing_category_rows_are_reported_after_runner_skips_one() -> None:
    expected = ["choice_checks", "numeric_checks"]
    assert runner._missing_category_rows(
        expected,
        [("choice_checks", {"id": "saved"})],
        [],
    ) == ["numeric_checks"]
    assert runner._missing_category_rows(
        expected,
        [("choice_checks", {"id": "submitted"})],
        [("numeric_checks", "fallback.json")],
    ) == []


def test_outbox_is_written_before_submission_and_survives_timeout(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    payload = {"model": "model-a", "prompt_category": "category-a"}
    path = runner._save_to_outbox(payload)

    saved = json.loads(path.read_text())
    assert saved["submission_id"] == payload["submission_id"]
    assert not list(tmp_path.glob("*.tmp"))

    def timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("acknowledgement lost")

    monkeypatch.setattr(runner.requests, "post", timeout)
    with pytest.raises(requests.exceptions.ReadTimeout):
        runner._post_outbox_file(path, "https://control.example", "secret", 120)

    assert path.exists()
    assert json.loads(path.read_text()) == saved


def test_outbox_filename_stays_bounded_for_long_run_identity(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    payload = {
        "experiment_tag": "synthetic-benchmark-grid-inference-host-example-model-long-identity-with-revision-0003-458ca7d14a47",
        "model": "fixture-owner/synthetic-model",
        "prompt_category": "fr_editorial_integrity",
        "concurrency": 10,
    }

    path = runner._save_to_outbox(payload)

    assert len(path.name.encode("utf-8")) < 100
    assert path.exists()
    assert json.loads(path.read_text()) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_outbox_write_uses_short_temporary_basename(tmp_path) -> None:
    path = tmp_path / f"{'x' * 238}.json"

    runner._atomic_write_json(path, {"saved": True})

    assert path.exists()
    assert json.loads(path.read_text()) == {"saved": True}
    assert len(path.name.encode("utf-8")) < 255
    assert not list(tmp_path.glob("*.tmp"))


def test_successful_submission_drains_older_outbox_entries(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    older = runner._save_to_outbox(
        {"model": "model-a", "prompt_category": "older"},
    )
    current = runner._save_to_outbox(
        {"model": "model-a", "prompt_category": "current"},
    )
    acknowledgements = iter(
        [_response(201, {"status": "created", "id": "current-id"}), _response(201, {"status": "created", "id": "older-id"})]
    )
    monkeypatch.setattr(
        runner.requests,
        "post",
        lambda *args, **kwargs: next(acknowledgements),
    )

    _, current_ack = runner._post_outbox_file(
        current, "https://control.example", "secret", 120
    )
    replayed, failures = runner._replay_fallbacks(
        "https://control.example", "secret", 120
    )

    assert current_ack["id"] == "current-id"
    assert [ack["id"] for _, ack in replayed] == ["older-id"]
    assert failures == 0
    assert not older.exists()
    assert not current.exists()
    assert not list(tmp_path.glob("fallback_*.json"))
    assert not list(tmp_path.glob("submitted_*.json"))


def test_replay_adds_and_persists_idempotency_to_legacy_fallback(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    path = tmp_path / "fallback_legacy.json"
    path.write_text(
        json.dumps(
            {
                "experiment_tag": "campaign-legacy",
                "model": "legacy-model",
                "prompt_category": "legacy",
            }
        )
    )

    def acknowledge(*args, **kwargs):
        on_disk = json.loads(path.read_text())
        assert on_disk["submission_id"] == kwargs["json"]["submission_id"]
        return _response(201, {"status": "created", "id": "legacy-id"})

    monkeypatch.setattr(runner.requests, "get", lambda *args, **kwargs: _response(200, []))
    monkeypatch.setattr(runner.requests, "post", acknowledge)
    replayed, failures = runner._replay_fallbacks(
        "https://control.example", "secret", 120
    )

    assert failures == 0
    assert replayed[0][1]["id"] == "legacy-id"


def test_replay_reconciles_a_legacy_fallback_already_in_history(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    payload = {
        "experiment_tag": "campaign-legacy",
        "model": "legacy-model",
        "prompt_category": "legacy-category",
        "mode": "warm",
        "concurrency": 10,
        "framework": "vllm",
    }
    path = tmp_path / "fallback_legacy.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        runner.requests,
        "get",
        lambda *args, **kwargs: _response(200, [{**payload, "id": "existing-id"}]),
    )

    def unexpected_post(*args, **kwargs):
        raise AssertionError("an already committed legacy fallback must not be posted")

    monkeypatch.setattr(runner.requests, "post", unexpected_post)
    replayed, failures = runner._replay_fallbacks(
        "https://control.example", "secret", 120
    )

    assert failures == 0
    assert replayed[0][1] == {"status": "already_created", "id": "existing-id"}
    assert not path.exists()

"""Check setup preservation and argument forwarding without pip or model downloads."""
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def checkout(tmp_path):
    root = tmp_path / "runner with spaces"
    (root / "scripts").mkdir(parents=True)
    for name in ("run.sh", "setup.sh"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (root / ".env.example").write_text("API_KEY=\n")
    (root / "requirements.txt").write_text("")
    return root


def fake_python(root, script):
    path = root / ".venv/bin/python"
    path.parent.mkdir(parents=True)
    path.write_text("#!/usr/bin/env bash\n" + script)
    path.chmod(0o700)
    return path


def test_launcher_reports_missing_installation(tmp_path):
    root = checkout(tmp_path)
    result = subprocess.run(["bash", str(root / "scripts/run.sh"), "--help"], capture_output=True, text=True)
    assert result.returncode == 1 and "setup.sh" in result.stderr


def test_launcher_preserves_arguments_exit_status_and_working_directory(tmp_path):
    root = checkout(tmp_path)
    fake_python(root, 'printf "%s\\n" "$PWD" "$@"\nexit 7\n')
    result = subprocess.run(["bash", str(root / "scripts/run.sh"), "fixture model", "--suite", "suite with spaces"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 7
    assert result.stdout.splitlines() == [str(root), "scripts/run_remote_benchmark.py", "fixture model", "--suite", "suite with spaces"]


def test_repeated_setup_preserves_private_configuration(tmp_path):
    root = checkout(tmp_path)
    fake_python(root, "exit 0\n")
    environment = dict(os.environ, PYTHON_BIN=str(root / ".venv/bin/python"))
    command = ["bash", str(root / "scripts/setup.sh")]
    subprocess.run(command, env=environment, check=True, capture_output=True)
    assert (root / ".env").stat().st_mode & 0o777 == 0o600
    (root / ".env").write_text("API_KEY=synthetic-local-value\n")
    result = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    assert (root / ".env").read_text() == "API_KEY=synthetic-local-value\n"
    assert "synthetic-local-value" not in result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("zip") is None, reason="zip is required to build archives")
def test_bundle_excludes_personal_files_even_inside_source_folders(tmp_path):
    root = checkout(tmp_path)
    for name in ("build_bundle.sh", "run_remote_benchmark.py"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    for name in ("README.md", "LICENSE"):
        (root / name).write_text("Synthetic packaging fixture\n")
    for folder in ("src", "benchmark_scoring"):
        (root / folder / "private").mkdir(parents=True)
        (root / folder / "__init__.py").write_text("")
        for name in (".env", "notes.md", "analysis.ipynb", "credential.key", "results.json"):
            (root / folder / name).write_text("synthetic private fixture")
        (root / folder / "private" / "helper.py").write_text("# Local experiment\n")
    (root / ".env").write_text("API_KEY=synthetic-private-value\n")
    subprocess.run(["bash", str(root / "scripts/build_bundle.sh")], check=True, capture_output=True)
    with zipfile.ZipFile(root / "dist/universal-benchmark-runner.zip") as archive:
        files = {name.removeprefix("universal-benchmark-runner/") for name in archive.namelist() if not name.endswith("/")}
    assert files == {
        "src/__init__.py", "benchmark_scoring/__init__.py", "scripts/run_remote_benchmark.py",
        "scripts/run.sh", "scripts/setup.sh", ".env.example", "requirements.txt", "README.md", "LICENSE", "REVISION",
    }

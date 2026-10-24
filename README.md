# LLM Bench Runner

## Deterministic scoring

`benchmark_scoring/registry.py` drives scorer execution and task validation. Each scorer registers its callable, description, configuration guidance and runnable reference example once. No AI judge, model download or BERTScore dependency is involved. Reference overlap and authored constraints measure their stated rules; they do not establish semantic truth or prose quality. Summary scoring defaults to ROUGE-L F1 ≥ 0.5. Ungraded generation diagnostics are retained for historical compatibility, outside the selectable scorer catalog.

To release a new scorer in both repositories, run `python scripts/sync_scoring.py --platform <platform-checkout>` and rebuild the platform. Use `--check` to verify package equality without writing. The API and task-editor dropdown derive their options from this package automatically; no duplicate UI list needs editing. Both repositories include their own copy so neither depends on a sibling checkout at runtime. The CLI compares implementation fingerprints with the platform before inference and requires matching scoring releases.

A portable CLI for running LLM benchmarks against Ollama, vLLM, and OpenAI-compatible inference endpoints.

The runner fetches suites, prompts, strategies, and runtime criteria from **LLM Bench Platform**, executes tasks beside your inference engine, records quality and performance measurements, and submits results to the platform API. It has no direct access to the platform database and does not require its source code.

## Install

Requires Python 3.12 or later and a reachable inference endpoint.

On the machine running Ollama or vLLM, clone the runner and install its dependencies. The platform repository is not required on this machine.

```sh
git clone https://github.com/aamohmd/LLM-Bench-Runner.git &&
  cd LLM-Bench-Runner &&
  bash scripts/setup.sh
bash scripts/run.sh --help
```

These are Bash commands for Linux/macOS, or WSL on Windows. Setup creates `.venv`, installs requirements, checks CLI startup, and creates a private `.env` only when absent. It preserves existing configuration and does not update your Git checkout. Set `PYTHON_BIN=python3.12` when your default Python is older. If the OS lacks venv support, install its Python venv package first. Use matching platform/runner releases; incompatible scorer fingerprints fail before inference.

Set `BACKEND_URL` and `API_KEY` in `.env`. Use a dedicated administrator key provisioned by the platform administrator; the current API requires administrator scope for benchmark execution. Create benchmark suites and prompts in the platform before running them. This repository includes no private datasets, campaign history, or model weights.

Edit `.env` without committing it. Use the platform address reachable from this machine; `localhost` means this machine. The dashboard URL can proxy API requests, or use a secured direct API address. Never use the platform's internal Docker hostname on a separate host. `bash scripts/run.sh` always uses this checkout's virtual environment and loads its `.env`; shell environment values take precedence. No activation is needed. It forwards all CLI flags and exit status.

Prompts and strategies must exist in the platform registry. The runner executes the composed text returned by the API; it has no built-in prompt or strategy catalog. A strategy with an empty directive executes the registered prompt unchanged.

## Run

From this repository root, set the model, suite, prompt, and strategy to values available in your inference runtime and platform:

```sh
bash scripts/run.sh "$MODEL" \
  --engine ollama \
  --suite "$SUITE" \
  --prompt "$PROMPT" \
  --strategy "$STRATEGY" \
  --mode warm --runs 1 --concurrency 1 \
  --experiment-tag "local-evaluation-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use `--help` for endpoint configuration, engine selection, trace capture, and output limits. Every real run needs a unique, descriptive experiment tag. Existing inference services and model installation are managed separately; optional dedicated vLLM execution is described below.

The normal flow is: start Ollama or vLLM with your model, select registered definitions in the dashboard, paste its command here, then refresh the dashboard to inspect results. The platform needs no inference engine. Running the runner beside the engine ensures local hardware telemetry describes the inference host; when connecting to a remote inference service, runner-local hardware is not evidence of that server's hardware.

## Runtime configuration

For Ollama and vLLM, `--temperature`, `--seed`, `--response-format text|json` and `--max-output-tokens` control actual requests. Defaults are temperature 0, seed 42 and text output. Seeds are reproducibility inputs, not guarantees. JSON mode requires JSON-compatible task instructions and scoring references. Decision mode retains its own temperature/seed policy and rejects conflicting overrides.

Ollama accepts `--ollama-base-url` and optional `--context-length`, sent as `num_ctx`. Omit the context limit for automatic task budgeting. Context settings that cannot fit the conservative task/output budget are rejected. Recorded context comes from `/api/ps`, never from substituting a requested value for an observation.

vLLM normally connects to `--vllm-base-url` or `VLLM_BASE_URL`. To configure a dedicated server, use the native CLI on a supported GPU host with vLLM installed in the same Python environment. Add `--launch-vllm` and an explicit unused local `--vllm-base-url http://localhost:<port>/v1`. Optional launch flags are `--vllm-max-model-len`, `--vllm-gpu-memory-utilization`, `--vllm-quantization`, `--vllm-kv-cache-dtype`, and `--vllm-prefix-caching enabled|disabled`. Blank/omitted settings use engine defaults; supported values depend on the installed vLLM, model and hardware. Prefix-cache flag spelling is discovered from the installed engine. Startup logs show compatibility failures; `--vllm-startup-timeout` controls the readiness deadline.

The runner refuses an occupied port, starts a loopback-only child process, waits for the requested model, then benchmarks and cleans up its own process group even on failure or Ctrl+C. It never restarts an external server. Dedicated launches use `--generation-config vllm` so model repository generation defaults do not override the baseline. Launch arguments are recorded with source `runner_launched_vllm`; they are configuration provenance, not measured VRAM. Unreported external-server settings remain unknown. Timing, token usage, engine-reported context and available memory/KV telemetry are recorded separately. Runtime settings are included in configuration identity to keep comparisons separate.

## Docker

```sh
docker compose build
docker compose run --rm runner --help
```

Pass the same CLI arguments after `runner` to execute a benchmark. Inside Docker, `localhost` refers to the runner container. The Compose example connects to services on the host through `host.docker.internal`; edit `.env` for remote services. The image is an HTTP client and does not install Ollama, vLLM, or model weights. For measurements of local GPU hardware, run the native CLI on the inference host; container hardware visibility may differ.

Tasks must include their required source context in the registered prompt. The runner executes those tasks directly; it does not maintain a local retrieval corpus, vector database, or reranker. Tasks that explicitly request external retrieval are rejected before inference.

## Packaging

Create a portable archive with `bash scripts/build_bundle.sh`. The archive contains runner source, core requirements, and revision information; it excludes private configuration, datasets, and infrastructure code.

The runtime archive includes only Python modules from `src/` and `benchmark_scoring/` plus the explicit CLI, setup/launcher scripts, environment template, requirements, README, license and revision marker. Personal notes, notebooks, reports, credentials and local results stay ignored in the checkout. Synthetic tests, development requirements and CI remain public for contributors, outside the runtime archive and image. Review `git status` and the diff before publishing: ignore rules do not remove already tracked files or earlier Git history.

## Dependencies

`requirements.txt` lists the Python libraries needed by the CLI.

## Tests and CI

Install development dependencies and run the offline regression suite:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Tests use synthetic tasks, mocked engines and temporary files. Live HTTP requests are blocked. They cover every selectable scorer's reference example, malformed tasks, engine streaming and token/timing provenance, missing/empty answers, reasoning, output limits, generation timeouts, retries, concurrency, repeated scoring, CLI validation, cancellation, vLLM process ownership, setup preservation, and durable submission/replay. Test files are public; private configuration, datasets and generated results are excluded. Tests are not included in the runtime archive or Docker image.

CI runs the suite on Linux with Python 3.12, 3.13 and 3.14, and macOS with Python 3.12. It checks dependencies and compilation, retains JUnit reports, and separately verifies native setup, the launcher, an independently extracted archive, Docker Compose and the container entry point. Jobs have time limits and require no API keys, GPU, model downloads or running platform. These protocol tests do not replace an actual GPU vLLM benchmark.

For runtime verification, use a small model already installed in Ollama and a small synthetic suite on a local platform: one repetition, concurrency 1, a unique experiment tag, and the normal `bash scripts/run.sh` entry point. Check non-empty answers, task scores, token/timing telemetry and recorded results. No model should be pulled automatically for testing.

## Failure recovery

Completed category results are written atomically to a private local outbox before submission. The runner removes a file only after the platform acknowledges a stored run ID with `created` or `already_created`. Authentication failures, redirects, malformed responses, timeouts and ambiguous acknowledgements leave the file intact. Replays reuse the same submission ID to avoid duplicate records; corrupt entries remain on disk while other entries are attempted.

After fixing platform access or availability, replay from the same runner checkout to the intended platform installation:

```sh
bash scripts/run.sh --replay-fallbacks
```

Keep `BACKEND_URL` and `API_KEY` configured in `.env`. Exit code 0 means completion; 4 means results remain for replay; 6 means categories lack a submitted or saved result. Codes 1–3 indicate configuration/runtime, argument/context, or repeated engine failure; 5 indicates exhaustion of the completion budget for every probe task. Ctrl+C or a cancelled prompt exits 130. Completed outbox entries survive interruptions; an unfinished category may need rerunning. The runner stops only a vLLM process it launched and leaves existing services running. Non-interactive sessions must supply the required flags instead of waiting for terminal prompts.

## Contributing

Open an issue or pull request with reproduction and validation steps. Keep model selection and platform registries configurable. Do not commit credentials, private datasets, model weights, or generated results.

## License

Apache-2.0. Third-party dependencies, models, and datasets retain their own licenses.

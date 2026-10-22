# LLM Bench Runner

## Deterministic scoring

`benchmark_scoring/registry.py` drives scorer execution and task validation. Each scorer registers its callable, description, configuration guidance and runnable reference example once. No AI judge, model download or BERTScore dependency is involved. Reference overlap and authored constraints measure their stated rules; they do not establish semantic truth or prose quality. Summary scoring defaults to ROUGE-L F1 ≥ 0.5. Ungraded generation diagnostics are retained for historical compatibility, outside the selectable scorer catalog.

To release a new scorer in both repositories, run `python scripts/sync_scoring.py --platform <platform-checkout>` and rebuild the platform. Use `--check` to verify package equality without writing. The API and task-editor dropdown derive their options from this package automatically; no duplicate UI list needs editing. Both repositories include their own copy so neither depends on a sibling checkout at runtime. The CLI compares implementation fingerprints with the platform before inference and requires matching scoring releases.

A portable CLI for running LLM benchmarks against Ollama, vLLM, and OpenAI-compatible inference endpoints.

The runner fetches suites, prompts, strategies, and runtime criteria from **LLM Bench Platform**, executes tasks beside your inference engine, records quality and performance measurements, and submits results to the platform API. It has no direct access to the platform database and does not require its source code.

## Install

Requires Python 3.12 or later and a reachable inference endpoint.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
cp .env.example .env
.venv/bin/python scripts/run_remote_benchmark.py --help
```

Set `BACKEND_URL` and `API_KEY` in `.env`. Use a dedicated administrator key provisioned by the platform administrator; the current API requires administrator scope for benchmark execution. Create benchmark suites and prompts in the platform before running them. This repository includes no private datasets, campaign history, or model weights.

Keep the virtual environment active when running commands copied from the dashboard.

Prompts and strategies must exist in the platform registry. The runner executes the composed text returned by the API; it has no built-in prompt or strategy catalog. A strategy with an empty directive executes the registered prompt unchanged.

## Run

From this repository root, set the model, suite, prompt, and strategy to values available in your inference runtime and platform:

```sh
.venv/bin/python scripts/run_remote_benchmark.py "$MODEL" \
  --engine ollama \
  --suite "$SUITE" \
  --prompt "$PROMPT" \
  --strategy "$STRATEGY" \
  --mode warm --runs 1 --concurrency 1 \
  --experiment-tag "local-evaluation-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use `--help` for endpoint configuration, engine selection, trace capture, and output limits. Every real run needs a unique, descriptive experiment tag. Inference engines and model downloads are managed separately from the CLI.

## Docker

```sh
docker compose build
docker compose run --rm runner --help
```

Pass the same CLI arguments after `runner` to execute a benchmark. Inside Docker, `localhost` refers to the runner container. The Compose example connects to services on the host through `host.docker.internal`; edit `.env` for remote services. The image is an HTTP client and does not install Ollama, vLLM, or model weights. For measurements of local GPU hardware, run the native CLI on the inference host; container hardware visibility may differ.

Tasks must include their required source context in the registered prompt. The runner executes those tasks directly; it does not maintain a local retrieval corpus, vector database, or reranker. Tasks that explicitly request external retrieval are rejected before inference.

## Packaging

Create a portable archive with `bash scripts/build_bundle.sh`. The archive contains runner source, core requirements, and revision information; it excludes private configuration, datasets, and infrastructure code.

## Dependencies

`requirements.txt` lists the Python libraries needed by the CLI.

## Contributing

Open an issue or pull request with reproduction and validation steps. Keep model selection and platform registries configurable. Do not commit credentials, private datasets, model weights, or generated results.

## License

Apache-2.0. Third-party dependencies, models, and datasets retain their own licenses.

# Cost-Optimizing LLM Router

A benchmark harness that measures **which optimizations reduce the cost of solving a problem set**. A small, locally-hosted model routes each problem to the cheapest external model likely to solve it; a separate server solves, validates, escalates on failure, and records all metrics.

- System overview: [`docs/architecture.md`](../docs/architecture.md)
- Task breakdown: [`docs/TODO.md`](../docs/TODO.md)
- Live progress board: [`PROGRESS.md`](PROGRESS.md)

## Install

Requires **Python 3.12**.

```bash
./install.sh mac     # local dev on Apple Silicon (vllm-metal)
./install.sh colab   # Colab / TPU (vllm-tpu)
```

## Configure

The resolution server calls external models through an OpenAI-compatible provider (OpenRouter by default). Set a key in `server/.env` or your shell:

```bash
export API_TOKEN=sk-...            # or OPENROUTER_API_KEY / LITELLM_API_KEY
# optional override:
export EXTERNAL_MODEL_BASE_URL=https://openrouter.ai/api/v1
```

Models, benchmark scores, and prompts live in [`configs/`](configs/).

## Run a benchmark

A run is identified by a `run_id`. Each run has a row of optimization flags that the server treats as the source of truth.

```bash
# 1. Register the run + its optimizations (pre-populates the optimizations table)
python server/metrics/create_run.py --run-id baseline_001 --baseline

# 2. Start the vLLM router + FastAPI resolution server
./run.sh
#   DTYPE=fp8 ./run.sh        # quantized router
#   ./run.sh --help           # all env vars / ports

# 3. Drive the problem set through the router (separate terminal)
python local-inference/main.py --run-id baseline_001 --server-url http://localhost:8001
#   --limit 3                 # smoke test on the first few problems
#   --problem-id 0            # run a single problem deterministically
#   --prompt-type capabilities  # router prompt that includes benchmark scores

# 4. Generate analysis tables + cost plot
python server/metrics/analysis_script.py
```

To compare an optimization against the baseline, register another run with different flags and repeat:

```bash
python server/metrics/create_run.py --run-id caveman_001 --caveman
python local-inference/main.py --run-id caveman_001 --server-url http://localhost:8001
```

Available flags (see `create_run.py --help`): `--baseline`, `--caveman`, `--capabilities-prompt`,
`--web-search-compression`, `--local-model-solves`, `--quantized-local-lm`, `--quantized-kv-cache`,
`--long-context-compression-lemma`, `--long-context-compression-ai`.

## End-to-end smoke test (no GPU / no API key)

Verify the whole resolution pipeline deterministically. This drives the real
FastAPI server in-process and mocks only the router + solver model calls (the
solver returns each problem's known-good answer, so validation/cost/metrics are
all exercised):

```bash
python scripts/e2e_smoke.py                       # a few problems from every domain
python scripts/e2e_smoke.py --domain math         # one domain
python scripts/e2e_smoke.py --per-domain 5
python scripts/e2e_smoke.py --domain code --per-domain 500 --analyze   # all code + analysis report
```

It exercises routing log -> resolution loop -> validation (match/tests/heuristic/judge)
-> cost function -> SQLite metrics, and asserts rows landed and problems were solved.
`--analyze` then runs the metrics analysis over the populated DB. Exit code is
non-zero on failure, so it doubles as a CI check (run on every PR; see
`.github/workflows/e2e-smoke.yml`).

## Live end-to-end eval (real models, needs an API key)

To measure how often the **real** external models solve the problems — and at
what cost — run the live eval. It uses the real resolution pipeline and real
model calls; only the router is a deterministic difficulty→model stand-in
(the real router needs the local vLLM model on a GPU):

```bash
export API_TOKEN=sk-or-...
python scripts/live_eval.py --domain math --per-domain 5 --analyze
python scripts/live_eval.py --domain code --per-domain 10
python scripts/live_eval.py --domain math --per-domain 5 --model openai/gpt-oss-120b:free
python scripts/live_eval.py --domain math --per-domain 5 --optimizations caveman
```

Results persist to a SQLite DB (path printed at the end) so you can re-run
`analysis_script.py` against it later.

## Problem domains

Problems are organized **by domain** so you can benchmark and compare per domain.
Built datasets live in `local-inference/problems/domains/<domain>.json`
(`code`, `math`, `reasoning`, `factual`, `image`). Run a single domain with `--domain`:

```bash
python local-inference/main.py --run-id code_001 --domain code
python local-inference/main.py --run-id math_001 --domain math
python local-inference/main.py --run-id all_001               # all domains (default)
```

## Adding problems

The domain files are **generated** by `build.py` from the vendored sources in
`local-inference/problems/datasets/` (see that folder's README). Edit a source
(e.g. `datasets/curated_noncode.json`) then rebuild:

```bash
python local-inference/problems/build.py
```

The loader **validates the loaded set at startup and fails loudly** on any malformed entry.
A problem's `category` determines its domain file.

| Field | Required | Notes |
|-------|----------|-------|
| `problem_id` | yes | Unique integer. |
| `problem` | yes | Non-empty prompt text. |
| `verify` | yes | `match` \| `tests` \| `judge` \| `heuristic`. |
| `difficulty` | yes | `very_easy` \| `easy` \| `medium` \| `hard` \| `very_hard`. |
| `category` | yes | e.g. `math`, `code`, `factual`, `reasoning`, `web`, `image`. |
| `answer` | if `match`/`judge` | Reference answer. |
| `assert_cases` | if `tests` | Python asserts run against the model's returned code. |
| `validator` | if `heuristic` | Name registered in [`server/validation/registry.py`](server/validation/registry.py). |
| `source_url` | optional | For `category: web`. |
| `image_url` | optional | For `category: image` (vision tier). |

Extra reference fields are ignored by the loader and the router. The imported HumanEval
problems carry `source`, `source_task_id`, `entry_point`, and `canonical_solution` for traceability.

The set (248 problems) is assembled by `build.py` from vendored datasets and grouped
into per-domain files. See
[`local-inference/problems/datasets/README.md`](local-inference/problems/datasets/README.md).
Current domains: code 167, math 50, reasoning 16, factual 10, image 5.

**Validation modes**
- `match` — exact string compare against `answer`.
- `tests` — execute the model's code with `assert_cases`; pass if no assertion fails.
- `judge` — an LLM grades the answer against `answer`.
- `heuristic` — a named function in the registry decides pass/fail (use for open-ended outputs).

To add a heuristic validator: write a `validate_*(model_answer, expected, ...) -> bool` in
`server/validation/validation_functions.py`, register it under a name in `_BY_NAME` in
`server/validation/registry.py`, and reference that name via the problem's `validator` field.

## Service URLs (defaults)

| Service | URL |
|---------|-----|
| vLLM router | `http://127.0.0.1:7654/v1` |
| FastAPI server | `http://127.0.0.1:8001` |
| API docs | `http://127.0.0.1:8001/docs` |
| Health | `http://127.0.0.1:8001/health` |
| Metrics | `http://127.0.0.1:8001/metrics` |

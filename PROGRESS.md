# Project Progress Tracker

Shared status board for the team. Update your rows when you start/finish work.
Full task breakdown lives in `../docs/TODO.md`; the system overview is in `../docs/architecture.md`.

**Status:** `not started` · `in progress` · `blocked` · `done`

---

## Milestone 1 — "It actually runs end-to-end" (P0)

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Problem-set JSON schema + loud loader-time validator | _ | done | `local-inference/utils.py::validate_problem_set`; rejects malformed problems at load. |
| Validation scaling (named validators, not per-`problem_id`) | _ | done | `verify: heuristic` now uses a `validator` name; registry is name-keyed with legacy-id fallback. |
| Grow problem set to 200+ (3 super-easy / 10 easy / 30 medium / 7 hard per person) | everyone | done | **248 problems, split by domain** into `problems/domains/<domain>.json`: code 167, math 50, reasoning 16, factual 10, image 5. Built by `problems/build.py` from vendored sources. Run one domain via `main.py --domain <name>`. |
| README run recipe | _ | done | Repo-root `README.md`. |
| End-to-end smoke test | _ | done | `python scripts/e2e_smoke.py` drives the real resolution server in-process (TestClient) over `/solve`, mocking only the router + solver model calls. Verifies routing -> resolution -> validation -> cost -> metrics. `--analyze` then runs the metrics analysis over the populated DB. Confirmed: **code 164/164, math 50/50** solve E2E. CI-friendly; no GPU/API key needed. |

### Locked problem schema (Milestone 1)

Each problem entry (in `local-inference/problems/datasets/*` sources, built into `problems/domains/<domain>.json`):

| Field | Required | Notes |
|-------|----------|-------|
| `problem_id` | yes | Unique integer. |
| `problem` | yes | Non-empty prompt text. |
| `verify` | yes | One of `match`, `tests`, `judge`, `heuristic`. |
| `difficulty` | yes | One of `very_easy`, `easy`, `medium`, `hard`, `very_hard`. |
| `category` | yes | e.g. `math`, `code`, `factual`, `reasoning`, `web`, `image`. |
| `answer` | if `match`/`judge` | Reference answer. |
| `assert_cases` | if `tests` | Python asserts run against the model's code. |
| `validator` | if `heuristic` | Name of a validator registered in `server/validation/registry.py`. |
| `source_url` | optional | For `category: web` problems. |
| `image_url` | optional | For `category: image` problems (vision tier, M4). |

> The loader **fails loudly** with a list of every offending problem, so a bad entry can't silently corrupt a run.

---

## Milestone 2 — Optimizations are real (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Wire `long_context_compression_*` into solve path | Tanmay | done | Wired into `build_solver_messages`, gated on `long_context_compression_ai` / `_lemma` flags (`_ai` precedes `_lemma`); no-op under the char threshold, skipped for `tests` prompts. **Savings measured**: `long_context_original_chars` / `long_context_compressed_chars` on `SolveResponse` + `problem_solving`, surfaced as `/metrics.long_context_chars_saved`. |
| `quantized_local_lm` drives `run.sh` dtype | George | partial | `run.sh` maps the optimization to `DTYPE` (off→bf16, on→fp8) + documents it. Needs a GPU run to validate/record. |
| `quantized_kv_cache` made conditional | George | done | `--kv-cache-dtype fp8` now gated on `QUANTIZE_KV_CACHE` (default true); set false for a fair baseline. |
| `caveman` / `web_search_compression` savings quotable | Tanmay | done | Via `baseline_comparison` (`completion_tokens_delta`) + `web_context_chars_saved` + `token_breakdown`. |

## Milestone 3 — Tools & escalation honest (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Solver actually calls `run_python_code` | Tanmay | done | `verify: tests` now runs the candidate in the sandbox and feeds failures back (solve→run→repair). `tool-server/core.py` split out so the runner works without `fastmcp` (CI-safe). |
| Model-native vs deterministic tool calling | Tanmay | done | Decided deterministic/server-driven; documented in `architecture.md` + docstring. |
| Tool server as standalone FastMCP service | _ | partial | Executors split into `core.py`; `server.py` registers them via FastMCP. Networked deployment + client still TODO. |
| Server-side queue / meaningful `/complete` | _ | not started | |

## Milestone 4 — Vision tier (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Image problems + image-token cost | George/Tanmay | done | `image_url` is a first-class schema field (passed through by local-inference); `cost_function` prices image tokens at `IMAGE_TOKEN_PREMIUM`× with `estimate_image_tokens`. |
| VLM-describe routing | _ | not started | Schema + cost ready; needs the VLM call wired into the solve path. |

## Milestone 5 — Metrics tell the story (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Router calibration (over/under-routing) | Tanmay | done | `router_calibration()` per-run under/over-routed counts + rates. |
| Cost-vs-solve-rate frontier chart | Tanmay | done | `cost_vs_solve_rate_frontier()` + `plot_cost_vs_solve_rate()` (`outputs/cost_vs_solve_rate.png`). |
| Token breakdown + reasoning audit | Tanmay | done | `token_breakdown()` and `reasoning_audit()` added to the report + CSVs. |

## Milestone 6 — Reproducible & shippable (P2)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| HumanEval / lm-eval-harness integration | Tanmay | not started | |
| `handle_solution` dead path | Tanmay | done | Removed; `handle` only accepts `"problem"`. |
| Branch consolidation | _ | not started | |
| CI smoke run | _ | done | `.github/workflows/e2e-smoke.yml` runs `scripts/e2e_smoke.py --per-domain 500 --analyze` on every PR (and pushes to `main`) — all 245 problems through the pipeline + the analysis report — installing only `requirements/ci.txt` (no GPU/API key). |

---

## Changelog
- **2026-06-10** — Added a **live end-to-end eval** (`scripts/live_eval.py`): drives the real resolution pipeline against real external models (no oracle mocking; needs `API_TOKEN`), with a deterministic router stand-in (real router needs the GPU vLLM model). Supports `--domain` / `--per-domain` / `--model` (force a solver) / `--optimizations` (enable flags) / `--analyze`, and persists a metrics DB for later analysis. Verified the plumbing without network (missing-key guard, per-problem flow, optimization seeding, analyze) via a patched-model control-flow test.
- **2026-06-10** — Verified the **imported code + math sets solve end-to-end and analysis runs** on the result. Ran every imported problem through the pipeline under the oracle solver: **code 164/164** and **math 50/50** solved (`numeric_match` correctly reads `#### N` from GSM8K canonical solutions). Added `scripts/e2e_smoke.py --analyze`, which runs the metrics analysis (run summary, baseline comparison, model usage, cost-by-difficulty, cost-vs-solve-rate, token breakdown, router calibration) over the populated DB. Wired into CI (`requirements/ci.txt` gains `pandas`; the workflow now runs `--per-domain 500 --analyze`, exercising all 245 problems). Note: vision/image is parked as non-essential per scope.
- **2026-06-10** — Broad milestone push across M2–M6:
  - **M3 code self-verify:** `verify: tests` now runs the model's candidate against its asserts in the sandboxed runner and feeds failures back for a solve→run→repair loop. Split `tool-server/core.py` (plain executors, no `fastmcp`) from the FastMCP `server.py` so the runner — and CI — work without the MCP stack. Documented the deterministic (server-driven) tool-calling decision.
  - **M4 vision foundations:** `image_url` is now a first-class schema field (passed through by local-inference); `cost_function` prices image tokens at a premium (`IMAGE_TOKEN_PREMIUM`, `estimate_image_tokens`, `image_tokens` on `CompletionConfig`).
  - **M5 metrics:** added `cost_vs_solve_rate_frontier` (+ scatter plot), `token_breakdown`, `router_calibration` (over/under-routing), and `reasoning_audit` to `analysis_script`, wired into the report + CSV exports.
  - **M2 quantization flags:** `run.sh` now gates `--kv-cache-dtype fp8` on `QUANTIZE_KV_CACHE` and maps `quantized_local_lm`↔`DTYPE`, documented in `--help`.
  - **M6 + quick wins:** removed the dead `handle_solution` path; `STRONGEST_MODEL_ID` derives from `models.yaml`; `create_run.py --help` seeds the baseline step; tidied `prompts.yaml`; fixed `install.sh` usage string.
  - Verified with targeted in-process checks (repair loop, image cost premium, analysis functions against a synthetic DB) and the E2E smoke test (still 15/15). `local_model_solves` deepening was deferred (needs GPU + an honest validate round-trip rather than echoing the reference answer).
- **2026-06-10** — Made long-context compression **measurable**: added `long_context_original_chars` / `long_context_compressed_chars` to `SolveResponse` and the `problem_solving` table (with an MVP-safe `ALTER TABLE` migration mirroring the `web_context_*` pattern), threaded them out of `build_solver_messages`, and exposed `long_context_chars_saved` on `/metrics`. The smoke test now prints chars saved. Verified end-to-end: a compression-enabled run recorded 21,032 → 8,043 chars (12,989 saved) in both the response and `/metrics`; baseline reports 0.
- **2026-06-10** — Milestone 2: wired `long_context_compression_lemma` / `long_context_compression_ai` into the solve path. `build_solver_messages` now compresses the solver's user content when the run flag is set (`_ai` takes precedence over `_lemma`); both no-op below the ~8k-char threshold and are skipped for whitespace-sensitive `tests` prompts. Verified the gating (30,041→8,043 chars when enabled, untouched at baseline / in `tests` mode) and confirmed the E2E smoke test still passes 15/15.
- **2026-06-09** — Wired the E2E smoke test into CI: `.github/workflows/e2e-smoke.yml` runs `scripts/e2e_smoke.py` on every pull request, on pushes to `main`, and on manual dispatch. Uses Python 3.12 and a minimal `requirements/ci.txt` (fastapi, httpx, openai, pyyaml — no ML stack), since the test mocks model calls and reads the committed domain datasets. Verified: clean install + run from public PyPI passes 15/15.
- **2026-06-09** — Added `scripts/e2e_smoke.py`: in-process E2E test that drives the real FastAPI resolution server over `/solve` (routing -> resolution -> validation -> cost -> SQLite metrics), mocking only the router and solver model calls (oracle returns each problem's known-good answer). Runs per-domain or across all; no GPU/API key required.
- **2026-06-09** — Split problems **by domain** (`problems/domains/<domain>.json`) so runs can target one domain (`main.py --domain code|math|reasoning|factual|image`). Added `problems/build.py` (assembles domains from vendored sources, grouped by `category`); the dataset importers are now vendor-only modules; removed the combined `problems.json`.
- **2026-06-09** — Expanded math/logic → **248 total**: GSM8K slice 15→40, plus 9 curated arithmetic (`math`) and 10 curated logic/sequence puzzles (`reasoning`). All `numeric_match`/`text_equals_ci`, answers hand-verified.
- **2026-06-09** — Added ~33 non-code problems → **204 total**. GSM8K math slice (15, vendored, `numeric_match`) + 18 curated factual/reasoning/image. New validators `numeric_match` and `text_equals_ci`. Image problems use stable Wikimedia flag URLs (staged for the M4 vision tier). All three importers (`import_humaneval`, `import_gsm8k`, `import_curated`) are idempotent and compose.
- **2026-06-09** — Vendored the raw HumanEval dataset at `local-inference/problems/datasets/humaneval.json` (~214 KB, MIT) so `import_humaneval.py` runs offline; `--refresh` re-downloads.
- **2026-06-09** — Imported 164 HumanEval problems (171 total). IDs `1000–1163`; difficulty by reference-solution length (≈57 easy / 72 medium / 35 hard); each carries `source`, `source_task_id`, `entry_point`, and `canonical_solution` for reference. All canonical solutions verified against our validator before import.
- **2026-06-09** — Milestone 1 kickoff: locked problem schema, added loader-time validator, refactored validation to scale past 7 problems, added README.

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
| End-to-end smoke test | _ | done | `python scripts/e2e_smoke.py` drives the real resolution server in-process (TestClient) over `/solve`, mocking only the router + solver model calls. Verifies routing -> resolution -> validation -> cost -> metrics. CI-friendly; no GPU/API key needed. |

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
| Wire `long_context_compression_*` into solve path | _ | not started | Defined in `optimizations.py`, never called. |
| `quantized_local_lm` drives `run.sh` dtype | George | not started | |
| `quantized_kv_cache` made conditional | George | not started | Currently always `--kv-cache-dtype fp8`. |

## Milestone 3 — Tools & escalation honest (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Solver actually calls `run_python_code` | _ | not started | |
| Tool server as standalone FastMCP service | _ | not started | |
| Server-side queue / meaningful `/complete` | _ | not started | |

## Milestone 4 — Vision tier (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Image problems + image-token cost | George | not started | |
| VLM-describe routing | _ | not started | |

## Milestone 5 — Metrics tell the story (P1)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Router calibration (over/under-routing) | George | not started | |
| Cost-vs-solve-rate frontier chart | _ | not started | |

## Milestone 6 — Reproducible & shippable (P2)
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| HumanEval / lm-eval-harness integration | Tanmay | not started | |
| Branch consolidation | _ | not started | |
| CI smoke run | _ | done | `.github/workflows/e2e-smoke.yml` runs `scripts/e2e_smoke.py` on every PR (and pushes to `main`), installing only `requirements/ci.txt` (no GPU/API key). |

---

## Changelog
- **2026-06-09** — Wired the E2E smoke test into CI: `.github/workflows/e2e-smoke.yml` runs `scripts/e2e_smoke.py` on every pull request, on pushes to `main`, and on manual dispatch. Uses Python 3.12 and a minimal `requirements/ci.txt` (fastapi, httpx, openai, pyyaml — no ML stack), since the test mocks model calls and reads the committed domain datasets. Verified: clean install + run from public PyPI passes 15/15.
- **2026-06-09** — Added `scripts/e2e_smoke.py`: in-process E2E test that drives the real FastAPI resolution server over `/solve` (routing -> resolution -> validation -> cost -> SQLite metrics), mocking only the router and solver model calls (oracle returns each problem's known-good answer). Runs per-domain or across all; no GPU/API key required.
- **2026-06-09** — Split problems **by domain** (`problems/domains/<domain>.json`) so runs can target one domain (`main.py --domain code|math|reasoning|factual|image`). Added `problems/build.py` (assembles domains from vendored sources, grouped by `category`); the dataset importers are now vendor-only modules; removed the combined `problems.json`.
- **2026-06-09** — Expanded math/logic → **248 total**: GSM8K slice 15→40, plus 9 curated arithmetic (`math`) and 10 curated logic/sequence puzzles (`reasoning`). All `numeric_match`/`text_equals_ci`, answers hand-verified.
- **2026-06-09** — Added ~33 non-code problems → **204 total**. GSM8K math slice (15, vendored, `numeric_match`) + 18 curated factual/reasoning/image. New validators `numeric_match` and `text_equals_ci`. Image problems use stable Wikimedia flag URLs (staged for the M4 vision tier). All three importers (`import_humaneval`, `import_gsm8k`, `import_curated`) are idempotent and compose.
- **2026-06-09** — Vendored the raw HumanEval dataset at `local-inference/problems/datasets/humaneval.json` (~214 KB, MIT) so `import_humaneval.py` runs offline; `--refresh` re-downloads.
- **2026-06-09** — Imported 164 HumanEval problems (171 total). IDs `1000–1163`; difficulty by reference-solution length (≈57 easy / 72 medium / 35 hard); each carries `source`, `source_task_id`, `entry_point`, and `canonical_solution` for reference. All canonical solutions verified against our validator before import.
- **2026-06-09** — Milestone 1 kickoff: locked problem schema, added loader-time validator, refactored validation to scale past 7 problems, added README.

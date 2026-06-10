# Problem datasets

Problems are organized **by domain** so we can benchmark and compare per domain
(code vs. math vs. reasoning, etc.).

- `../domains/<domain>.json` — built, per-domain datasets actually loaded at run time
  (`code`, `math`, `reasoning`, `factual`, `image`). **Generated — do not hand-edit.**
- `datasets/*.json` (this folder) — the raw/vendored sources of truth.

Rebuild the domain files from the sources:

```bash
python local-inference/problems/build.py             # offline rebuild from vendored data
python local-inference/problems/build.py --refresh   # re-download datasets first, then rebuild
```

Each problem's `category` field decides which domain file it lands in.

Run a benchmark on one domain:

```bash
python local-inference/main.py --run-id code_001 --domain code
python local-inference/main.py --run-id math_001 --domain math
python local-inference/main.py --run-id all_001               # all domains (default)
```

## Sources

### `base.json`
- Team-authored original problems (math, code, factual, reasoning).

### `humaneval.json` -> `code`
- **Source:** [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (164 rows, `test` split)
- **License:** MIT
- **Fields:** `task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`
- Built as `verify: tests` problems; every canonical solution is verified against our `run_code` validator during the build.

### `gsm8k.json` -> `math`
- **Source:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (first `SLICE_SIZE` rows of `main`/`test`; edit `SLICE_SIZE` in `import_gsm8k.py`)
- **License:** MIT
- **Fields:** `question`, `answer` (gold value follows `####`)
- Built as `verify: heuristic` problems using the `numeric_match` validator.

### `curated_noncode.json` -> `math` / `reasoning` / `factual` / `image`
- Team-authored, verifiable problems. Image problems reference stable Wikimedia Commons flag URLs (`Special:FilePath`).
- This file is the source of truth; edit it and re-run `build.py`.

## Refreshing vendored copies

`build.py --refresh` re-downloads everything. To refresh a single dataset's vendored
copy without rebuilding:

```bash
python local-inference/problems/import_humaneval.py --refresh
python local-inference/problems/import_gsm8k.py --refresh
```

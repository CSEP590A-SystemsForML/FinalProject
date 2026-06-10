"""
HumanEval dataset source module for the `code` domain.

Source: https://huggingface.co/datasets/openai/openai_humaneval (MIT licensed)

Each HumanEval task becomes a `verify: tests` code problem:
- problem: the function stub/docstring, prefixed with a completion instruction.
- assert_cases: the dataset's `check(candidate)` test plus a call on the entry point,
  formatted so server/validation/utils.py::run_code can execute it directly.
- canonical_solution / source / entry_point: kept for traceability.

The raw dataset is vendored at datasets/humaneval.json so builds run offline.
`build.py` calls load_rows()/build_problems() to assemble the domain files;
running this module directly only refreshes the vendored copy (--refresh).

Usage:
    python local-inference/problems/import_humaneval.py --refresh   # re-download vendor
"""

import argparse
import json
import urllib.request
from pathlib import Path

SOURCE = "openai/openai_humaneval"
TOTAL_ROWS = 164
ID_OFFSET = 1000  # HumanEval/N -> problem_id 1000 + N
RAW_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "humaneval.json"
API = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={SOURCE}&config=openai_humaneval&split=test"
)


def download_rows() -> list[dict]:
    rows: list[dict] = []
    for offset in range(0, TOTAL_ROWS, 100):
        length = min(100, TOTAL_ROWS - offset)
        url = f"{API}&offset={offset}&length={length}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        rows.extend(r["row"] for r in data["rows"])
    return rows


def load_rows(refresh: bool = False) -> list[dict]:
    """Load the vendored dataset, or download + cache it when refreshing/missing."""
    if not refresh and RAW_DATASET_PATH.exists():
        print(f"reading vendored dataset: {RAW_DATASET_PATH}")
        return json.loads(RAW_DATASET_PATH.read_text())

    print(f"downloading dataset from {SOURCE}")
    rows = download_rows()
    RAW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATASET_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"cached {len(rows)} raw rows to {RAW_DATASET_PATH}")
    return rows


def difficulty_for(canonical_solution: str) -> str:
    """Heuristic proxy by reference-solution length; recalibrate vs. solve rates."""
    lines = [ln for ln in canonical_solution.splitlines() if ln.strip()]
    n = len(lines)
    if n <= 3:
        return "easy"
    if n <= 9:
        return "medium"
    return "hard"


def run_code(function: str, assert_cases: str) -> bool:
    """Mirror of server/validation/utils.py::run_code, used to verify imports."""
    if not function:
        return False
    try:
        env: dict = {}
        exec(f"{function}\n{assert_cases}", env)
        return True
    except Exception:
        return False


def build_problems(rows: list[dict]) -> list[dict]:
    problems = []
    failures = []
    for row in rows:
        task_id = row["task_id"]
        n = int(task_id.split("/")[1])
        prompt = row["prompt"]
        canonical_solution = row["canonical_solution"]
        entry_point = row["entry_point"]

        full_solution = prompt + canonical_solution
        assert_cases = f"{row['test']}\n\ncheck({entry_point})"

        if not run_code(full_solution, assert_cases):
            failures.append(task_id)
            continue

        problems.append({
            "problem_id": ID_OFFSET + n,
            "problem": (
                "Complete the following Python function. Return the entire "
                "function definition, including any needed imports.\n\n" + prompt
            ),
            "answer": None,
            "verify": "tests",
            "difficulty": difficulty_for(canonical_solution),
            "category": "code",
            "assert_cases": assert_cases,
            "source": SOURCE,
            "source_task_id": task_id,
            "entry_point": entry_point,
            "canonical_solution": full_solution,
        })

    if failures:
        raise RuntimeError(
            f"{len(failures)} canonical solutions failed validation: {failures}"
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor the HumanEval dataset.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the dataset from Hugging Face and update the vendored copy.",
    )
    args = parser.parse_args()

    rows = load_rows(refresh=args.refresh)
    problems = build_problems(rows)
    print(f"vendored {len(rows)} rows, built {len(problems)} problems. Run build.py to assemble domains.")


if __name__ == "__main__":
    main()

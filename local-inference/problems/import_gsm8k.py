"""
GSM8K dataset source module for the `math` domain.

Source: https://huggingface.co/datasets/openai/gsm8k (MIT licensed)

Each row becomes a `verify: heuristic` math problem using the `numeric_match`
validator, which compares the final number in the model's answer against the
GSM8K gold answer (the value after "####"). Robust to surrounding prose.

A fixed slice (first SLICE_SIZE test rows) is vendored at datasets/gsm8k.json so
builds run offline. `build.py` calls load_rows()/build_problems(); running this
module directly only refreshes the vendored copy (--refresh).

Usage:
    python local-inference/problems/import_gsm8k.py --refresh   # re-download vendor
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

SOURCE = "openai/gsm8k"
SLICE_SIZE = 40
ID_OFFSET = 2000
RAW_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "gsm8k.json"
API = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={SOURCE}&config=main&split=test"
)


def download_rows() -> list[dict]:
    url = f"{API}&offset=0&length={SLICE_SIZE}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return [r["row"] for r in data["rows"]]


def load_rows(refresh: bool = False) -> list[dict]:
    if not refresh and RAW_DATASET_PATH.exists():
        print(f"reading vendored dataset: {RAW_DATASET_PATH}")
        return json.loads(RAW_DATASET_PATH.read_text())

    print(f"downloading dataset from {SOURCE}")
    rows = download_rows()
    RAW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATASET_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"cached {len(rows)} raw rows to {RAW_DATASET_PATH}")
    return rows


def gold_answer(answer: str) -> str:
    """The GSM8K gold answer is the value after '####'."""
    final = answer.split("####")[-1].strip()
    return final.replace(",", "").replace("$", "")


def difficulty_for(answer: str) -> str:
    # GSM8K embeds each arithmetic step as <<...>>; use the step count as a proxy.
    steps = len(re.findall(r"<<", answer))
    if steps <= 2:
        return "easy"
    if steps <= 4:
        return "medium"
    return "hard"


def build_problems(rows: list[dict]) -> list[dict]:
    problems = []
    for i, row in enumerate(rows):
        problems.append({
            "problem_id": ID_OFFSET + i,
            "problem": row["question"],
            "answer": gold_answer(row["answer"]),
            "verify": "heuristic",
            "validator": "numeric_match",
            "difficulty": difficulty_for(row["answer"]),
            "category": "math",
            "source": SOURCE,
            "source_index": i,
            "canonical_solution": row["answer"],
        })
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor a GSM8K slice.")
    parser.add_argument("--refresh", action="store_true", help="Re-download the dataset slice.")
    args = parser.parse_args()

    rows = load_rows(refresh=args.refresh)
    problems = build_problems(rows)
    print(f"vendored {len(rows)} rows, built {len(problems)} problems. Run build.py to assemble domains.")


if __name__ == "__main__":
    main()

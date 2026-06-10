"""
Assemble per-domain problem datasets from the vendored sources.

Each problem's `category` determines its domain file, written to
`domains/<category>.json`. This lets us benchmark and compare by domain
(code vs. math vs. reasoning, etc.) instead of one mixed set.

Sources:
- datasets/base.json            (the original hand-authored problems)
- datasets/humaneval.json       (-> code; via import_humaneval)
- datasets/gsm8k.json           (-> math; via import_gsm8k)
- datasets/curated_noncode.json (-> math / reasoning / factual / image)

Usage:
    python local-inference/problems/build.py             # offline rebuild from vendored data
    python local-inference/problems/build.py --refresh   # re-download datasets, then rebuild
"""

import argparse
import json
from pathlib import Path

import import_gsm8k
import import_humaneval

HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE / "datasets"
DOMAINS_DIR = HERE / "domains"


def collect_problems(refresh: bool) -> list[dict]:
    problems: list[dict] = []

    problems += json.loads((DATASETS_DIR / "base.json").read_text())
    problems += import_humaneval.build_problems(import_humaneval.load_rows(refresh=refresh))
    problems += import_gsm8k.build_problems(import_gsm8k.load_rows(refresh=refresh))
    problems += json.loads((DATASETS_DIR / "curated_noncode.json").read_text())

    return problems


def group_by_domain(problems: list[dict]) -> dict[str, list[dict]]:
    domains: dict[str, list[dict]] = {}
    for p in problems:
        category = p.get("category")
        if not category:
            raise ValueError(f"problem_id={p.get('problem_id')} has no category")
        domains.setdefault(category, []).append(p)
    for rows in domains.values():
        rows.sort(key=lambda x: x["problem_id"])
    return domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-domain problem datasets.")
    parser.add_argument("--refresh", action="store_true", help="Re-download source datasets first.")
    args = parser.parse_args()

    problems = collect_problems(refresh=args.refresh)

    ids = [p["problem_id"] for p in problems]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate problem_id(s) across domains: {dupes}")

    domains = group_by_domain(problems)

    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale domain files so a removed category does not linger.
    for existing in DOMAINS_DIR.glob("*.json"):
        existing.unlink()

    for domain, rows in sorted(domains.items()):
        (DOMAINS_DIR / f"{domain}.json").write_text(json.dumps(rows, indent=2) + "\n")

    total = sum(len(r) for r in domains.values())
    print(f"built {total} problems across {len(domains)} domains -> {DOMAINS_DIR}")
    for domain, rows in sorted(domains.items()):
        print(f"  {domain:<10} {len(rows):>4}")


if __name__ == "__main__":
    main()

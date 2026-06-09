import argparse
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "metrics.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"


OPTIMIZATION_FLAGS = [
    "baseline",
    "caveman",
    "capabilities_prompt",
    "quantized_local_lm",
    "quantized_kv_cache",
    "web_search_compression",
    "local_model_solves",
    "long_context_compression_lemma",
    "long_context_compression_ai",
]


def initialize_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


def upsert_optimization_run(
    db_path: Path,
    run_id: str,
    label: str | None,
    description: str | None,
    flags: dict[str, bool],
) -> None:
    initialize_db(db_path)

    columns = ["run_id", "label", "description", *OPTIMIZATION_FLAGS]
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column != "run_id"]
    update_clause = ", ".join(f"{column} = excluded.{column}" for column in update_columns)

    values = [
        run_id,
        label,
        description,
        *(int(flags.get(flag, False)) for flag in OPTIMIZATION_FLAGS),
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO optimizations ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(run_id) DO UPDATE SET
                {update_clause}
            """,
            values,
        )
        conn.commit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or update one metrics optimization run row."
    )
    parser.add_argument("--run-id", required=True, help="Stable run id, e.g. baseline_001.")
    parser.add_argument("--label", default=None, help="Human-readable run label.")
    parser.add_argument("--description", default=None, help="Short run description.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite metrics DB path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success output.",
    )

    for flag in OPTIMIZATION_FLAGS:
        parser.add_argument(
            f"--{flag.replace('_', '-')}",
            action="store_true",
            help=f"Enable {flag}.",
        )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    flags = {flag: bool(getattr(args, flag)) for flag in OPTIMIZATION_FLAGS}
    upsert_optimization_run(
        db_path=args.db_path,
        run_id=args.run_id,
        label=args.label,
        description=args.description,
        flags=flags,
    )

    if not args.quiet:
        enabled = [flag for flag, value in flags.items() if value]
        enabled_text = ", ".join(enabled) if enabled else "none"
        print(f"Created/updated run_id={args.run_id}; optimizations={enabled_text}")


if __name__ == "__main__":
    main()
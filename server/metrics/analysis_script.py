import sqlite3
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "metrics.db"
OUTPUT_IMAGE = BASE_DIR / "cost_by_optimizations.png"

OPTIMIZATION_COLS = [
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


def get_db_connection(db_path: Path = DB_PATH):
    return sqlite3.connect(db_path)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _read_sql_or_empty(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame()


def _active_optimizations(row: pd.Series) -> str:
    active = [col for col in OPTIMIZATION_COLS if bool(row.get(col, False))]
    return ", ".join(active) if active else "none"


def run_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return run-level solve/cost summary.
    """

    if not _table_exists(conn, "problem_solving"):
        return pd.DataFrame()

    query = """
    SELECT
        run_id,
        COUNT(*) AS total_problems,
        SUM(CASE WHEN solved THEN 1 ELSE 0 END) AS solved_problems,
        SUM(total_cost) AS total_cost,
        SUM(attempts) AS total_attempts,
        AVG(attempts) AS avg_attempts,
        SUM(CASE WHEN escalated THEN 1 ELSE 0 END) AS escalations,
        SUM(prompt_tokens) AS prompt_tokens,
        SUM(completion_tokens) AS completion_tokens
    FROM problem_solving
    GROUP BY run_id
    ORDER BY run_id
    """
    df = _read_sql_or_empty(conn, query)
    if df.empty:
        return df

    df["solve_rate"] = df["solved_problems"] / df["total_problems"]
    df["cost_per_solved"] = df.apply(
        lambda row: row["total_cost"] / row["solved_problems"] if row["solved_problems"] else None,
        axis=1,
    )
    return df


def run_summary_with_optimizations(conn: sqlite3.Connection) -> pd.DataFrame:
    summary = run_summary(conn)
    if summary.empty or not _table_exists(conn, "optimizations"):
        return summary

    optimizations = _read_sql_or_empty(conn, "SELECT * FROM optimizations")
    if optimizations.empty:
        return summary

    df = summary.merge(optimizations, on="run_id", how="left")
    df["active_optimizations"] = df.apply(_active_optimizations, axis=1)
    df["run_label"] = df.apply(
        lambda row: row["label"] if "label" in row and pd.notna(row["label"]) and row["label"] else row["run_id"],
        axis=1,
    )
    return df


def model_usage_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "problem_solving"):
        return pd.DataFrame()

    query = """
    SELECT
        run_id,
        model_id,
        COUNT(*) AS count,
        SUM(total_cost) AS total_cost,
        SUM(CASE WHEN solved THEN 1 ELSE 0 END) AS solved_count
    FROM problem_solving
    GROUP BY run_id, model_id
    ORDER BY run_id, count DESC
    """
    return _read_sql_or_empty(conn, query)


def cost_by_difficulty(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "problem_solving") or not _table_exists(conn, "routing"):
        return pd.DataFrame()

    query = """
    SELECT
        ps.run_id,
        COALESCE(r.difficulty, 'unknown') AS difficulty,
        COUNT(*) AS total_problems,
        SUM(CASE WHEN ps.solved THEN 1 ELSE 0 END) AS solved_problems,
        SUM(ps.total_cost) AS total_cost,
        AVG(ps.attempts) AS avg_attempts
    FROM problem_solving ps
    LEFT JOIN routing r
        ON ps.run_id = r.run_id
       AND ps.problem_id = r.problem_id
    GROUP BY ps.run_id, COALESCE(r.difficulty, 'unknown')
    ORDER BY ps.run_id, total_cost DESC
    """
    return _read_sql_or_empty(conn, query)


def most_expensive_problems(conn: sqlite3.Connection, limit: int = 10) -> pd.DataFrame:
    if not _table_exists(conn, "problem_solving"):
        return pd.DataFrame()

    query = f"""
    SELECT
        ps.run_id,
        ps.problem_id,
        ps.model_id,
        ps.solved,
        ps.escalated,
        ps.attempts,
        ps.prompt_tokens,
        ps.completion_tokens,
        ps.total_cost,
        r.difficulty,
        r.category
    FROM problem_solving ps
    LEFT JOIN routing r
        ON ps.run_id = r.run_id
       AND ps.problem_id = r.problem_id
    ORDER BY ps.total_cost DESC
    LIMIT {int(limit)}
    """
    return _read_sql_or_empty(conn, query)


def calculate_costs_and_plot(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """
    Plot total cost per run using problem_solving.total_cost.
    """

    owns_connection = conn is None
    if conn is None:
        conn = get_db_connection()

    try:
        df = run_summary_with_optimizations(conn)
        if df.empty:
            print("No run data found. Cost plot skipped.")
            return df

        if plt is None:
            print("matplotlib not installed. Cost plot skipped.")
            return df

        if "active_optimizations" not in df.columns:
            df["active_optimizations"] = "unknown"
        if "run_label" not in df.columns:
            df["run_label"] = df["run_id"]

        df["label"] = df.apply(
            lambda row: f"{row['run_label']}\n({row['active_optimizations']})",
            axis=1,
        )

        plt.figure(figsize=(10, 6))
        bars = plt.bar(df["label"], df["total_cost"], color="skyblue", edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"${height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.title("Total Cost by Run")
        plt.xlabel("Run")
        plt.ylabel("Total Cost")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        plt.close()
        print(f"Cost plot saved to {OUTPUT_IMAGE}")
        return df
    finally:
        if owns_connection:
            conn.close()


def determine_outliers(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """
    Determine simple routing outliers.
    """

    owns_connection = conn is None
    if conn is None:
        conn = get_db_connection()

    try:
        if not _table_exists(conn, "routing"):
            return pd.DataFrame()

        df = _read_sql_or_empty(conn, "SELECT run_id, problem_id, model_id, difficulty FROM routing")
        if df.empty:
            return pd.DataFrame()

        outliers = []
        for _, row in df.iterrows():
            run_id = row["run_id"]
            problem_id = row["problem_id"]
            model_id = str(row["model_id"]).strip().lower()
            difficulty = str(row["difficulty"]).strip().lower()

            is_outlier = False
            reason = ""

            if difficulty not in ("very_easy", "easy") and "gpt-oss-20b" in model_id:
                is_outlier = True
                reason = "Non-easy problem routed to gpt-oss-20b"
            elif difficulty == "medium" and not ("gpt-oss-120b" in model_id or "deepseek-v4-flash" in model_id):
                is_outlier = True
                reason = "Medium problem not routed to gpt-oss-120b or deepseek-v4-flash"
            elif difficulty == "hard" and ("gpt-oss-20b" in model_id or "gpt-oss-120b" in model_id):
                is_outlier = True
                reason = "Hard problem routed to gpt-oss-20b or gpt-oss-120b"
            elif difficulty in ("very_hard", "very hard") and "kimi-k2.6" not in model_id:
                is_outlier = True
                reason = "Very hard problem not routed to kimi-k2.6"

            if is_outlier:
                outliers.append(
                    {
                        "run_id": run_id,
                        "problem_id": problem_id,
                        "model_id": row["model_id"],
                        "difficulty": row["difficulty"],
                        "reason": reason,
                    }
                )

        return pd.DataFrame(outliers)
    finally:
        if owns_connection:
            conn.close()


def print_compact_report() -> None:
    if not DB_PATH.exists():
        print("No metrics DB found.")
        return

    conn = get_db_connection()
    try:
        sections = [
            ("RUN SUMMARY", run_summary_with_optimizations(conn)),
            ("MODEL USAGE", model_usage_summary(conn)),
            ("COST BY DIFFICULTY", cost_by_difficulty(conn)),
            ("MOST EXPENSIVE PROBLEMS", most_expensive_problems(conn)),
            ("ROUTING OUTLIERS", determine_outliers(conn)),
        ]

        for title, df in sections:
            print(f"\n=== {title} ===")
            if df.empty:
                print("No data.")
            else:
                print(df.to_string(index=False))

        calculate_costs_and_plot(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    print_compact_report()
import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from server.interfaces import LocalSolveRequest, SolveRequest, SolveResponse
from server.resolution.resolution import solve_problem

app = FastAPI()


def get_db_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    return base_dir / "metrics" / "metrics.db"


def _ensure_problem_solving_columns(conn: sqlite3.Connection) -> None:
    """
    Apply tiny MVP-safe schema migrations for existing local metrics DBs.
    """

    rows = conn.execute("PRAGMA table_info(problem_solving)").fetchall()
    existing_columns = {row[1] for row in rows}

    if "final_answer" not in existing_columns:
        conn.execute("ALTER TABLE problem_solving ADD COLUMN final_answer TEXT")
    if "web_context_original_chars" not in existing_columns:
        conn.execute(
            "ALTER TABLE problem_solving ADD COLUMN web_context_original_chars INTEGER DEFAULT 0"
        )
    if "web_context_sent_chars" not in existing_columns:
        conn.execute(
            "ALTER TABLE problem_solving ADD COLUMN web_context_sent_chars INTEGER DEFAULT 0"
        )
    if "long_context_original_chars" not in existing_columns:
        conn.execute(
            "ALTER TABLE problem_solving ADD COLUMN long_context_original_chars INTEGER DEFAULT 0"
        )
    if "long_context_compressed_chars" not in existing_columns:
        conn.execute(
            "ALTER TABLE problem_solving ADD COLUMN long_context_compressed_chars INTEGER DEFAULT 0"
        )


def initialize_db() -> None:
    base_dir = Path(__file__).resolve().parent
    db_path = get_db_path()
    schema_path = base_dir / "metrics" / "schema.sql"

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        _ensure_problem_solving_columns(conn)
        conn.commit()
    finally:
        conn.close()


def log_routing_decision(solve_request: SolveRequest) -> None:
    """
    Server-owned routing metric logging.

    local-inference sends run_id, problem data, router-selected model_id, and
    router reasoning; the server records the routing metric here.
    """

    conn = sqlite3.connect(get_db_path())
    try:
        conn.execute(
            """
            INSERT INTO routing (
                run_id,
                problem_id,
                model_id,
                difficulty,
                category,
                reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                solve_request.run_id,
                solve_request.problem_id,
                solve_request.model_id,
                solve_request.difficulty,
                solve_request.category,
                solve_request.router_reasoning,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_problem_solving_result(solve_response: SolveResponse) -> None:
    """
    Server-owned problem-solving metric logging.
    """

    conn = sqlite3.connect(get_db_path())
    try:
        conn.execute(
            """
            INSERT INTO problem_solving (
                run_id,
                problem_id,
                attempts,
                num_tool_calls,
                tool_invocations,
                model_id,
                solved,
                escalated,
                prompt_tokens,
                completion_tokens,
                total_cost,
                final_answer,
                web_context_original_chars,
                web_context_sent_chars,
                long_context_original_chars,
                long_context_compressed_chars,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                solve_response.run_id,
                solve_response.problem_id,
                solve_response.attempts,
                solve_response.num_tool_calls,
                ",".join(solve_response.tool_invocations),
                solve_response.model_id,
                solve_response.solved,
                solve_response.escalated,
                solve_response.prompt_tokens,
                solve_response.completion_tokens,
                solve_response.total_cost,
                solve_response.final_answer,
                solve_response.web_context_original_chars,
                solve_response.web_context_sent_chars,
                solve_response.long_context_original_chars,
                solve_response.long_context_compressed_chars,
                solve_response.error,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_run_optimizations(run_id: str) -> dict | None:
    """
    Returns the pre-inserted optimization metadata for run_id, if present.

    The optimizations table is intentionally populated before a run starts, so
    this row is the server-side source of truth for what each run_id means.

    If no row exists, return None. The resolution layer treats None as all
    optimizations disabled, which is the safe baseline behavior for MVP runs.
    """

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM optimizations WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


@app.on_event("startup")
def on_startup():
    initialize_db()


@app.get("/health")
async def health():
    """
    Returns service health and verifies SQLite connectivity.
    """

    try:
        conn = sqlite3.connect(get_db_path())
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "database": "disconnected",
                "error": repr(e),
            },
        )

    return {"status": "ok", "database": "connected"}


@app.post("/solve", response_model=SolveResponse)
async def solve(solve_request: SolveRequest) -> SolveResponse:
    """
    Accepts a routed problem from local-inference, resolves it, and records
    server-owned metrics.

    MVP behavior: return a valid SolveResponse on internal failure so one bad
    problem does not crash an entire benchmark run.
    """

    try:
        log_routing_decision(solve_request)
        _run_optimizations = get_run_optimizations(solve_request.run_id)

        solve_response = await solve_problem(solve_request, _run_optimizations)
        log_problem_solving_result(solve_response)
        return solve_response
    except Exception:
        failure_response = SolveResponse(
            run_id=solve_request.run_id,
            problem_id=solve_request.problem_id,
            model_id=solve_request.model_id,
            solved=False,
            attempts=0,
            final_answer=None,
            num_tool_calls=0,
            tool_invocations=[],
            prompt_tokens=0,
            completion_tokens=0,
            total_cost=0.0,
            web_context_original_chars=0,
            web_context_sent_chars=0,
            escalated=False,
            error="Solve failed before completion.",
        )

        try:
            log_problem_solving_result(failure_response)
        except Exception:
            pass

        return failure_response


@app.post("/local-solve", response_model=SolveResponse)
async def local_solve(local_request: LocalSolveRequest) -> SolveResponse:
    """
    Records a problem solved locally by local-inference.

    This keeps metrics centralized in the server while allowing the
    local_model_solves optimization to avoid an external model call.
    """

    routing_request = SolveRequest(
        run_id=local_request.run_id,
        problem_id=local_request.problem_id,
        problem=local_request.problem,
        answer=local_request.answer,
        verify=local_request.verify,
        difficulty=local_request.difficulty,
        model_id=local_request.model_id,
        router_reasoning=local_request.router_reasoning,
        category=local_request.category,
    )
    log_routing_decision(routing_request)

    solve_response = SolveResponse(
        run_id=local_request.run_id,
        problem_id=local_request.problem_id,
        model_id=local_request.model_id,
        solved=True,
        attempts=1,
        final_answer=local_request.final_answer,
        num_tool_calls=0,
        tool_invocations=[],
        prompt_tokens=0,
        completion_tokens=0,
        total_cost=0.0,
        web_context_original_chars=0,
        web_context_sent_chars=0,
        escalated=False,
        error=None,
    )
    log_problem_solving_result(solve_response)
    return solve_response


@app.get("/complete")
async def complete():
    """
    Returns if all tasks submitted to the server have been resolved.

    MVP note: /solve is synchronous from the caller's perspective, so there are
    no background tasks yet.
    """

    return {"complete": True}


@app.get("/metrics")
async def metrics():
    """
    Returns basic MVP metric aggregates from the server-owned SQLite database.
    """

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        routing_count = conn.execute("SELECT COUNT(*) FROM routing").fetchone()[0]
        problem_solving_count = conn.execute("SELECT COUNT(*) FROM problem_solving").fetchone()[0]
        optimization_runs = conn.execute("SELECT COUNT(*) FROM optimizations").fetchone()[0]

        summary = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN solved THEN 1 ELSE 0 END), 0) AS solved_problems,
                COALESCE(SUM(total_cost), 0) AS total_cost,
                COALESCE(SUM(attempts), 0) AS total_attempts,
                COALESCE(SUM(CASE WHEN escalated THEN 1 ELSE 0 END), 0) AS escalations,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(long_context_original_chars), 0) AS long_context_original_chars,
                COALESCE(SUM(long_context_compressed_chars), 0) AS long_context_compressed_chars
            FROM problem_solving
            """
        ).fetchone()

        model_rows = conn.execute(
            """
            SELECT model_id, COUNT(*) AS count
            FROM problem_solving
            GROUP BY model_id
            ORDER BY count DESC
            """
        ).fetchall()

        solved_problems = summary["solved_problems"]
        solve_rate = solved_problems / problem_solving_count if problem_solving_count else 0

        return {
            "routing_count": routing_count,
            "problem_solving_count": problem_solving_count,
            "optimization_runs": optimization_runs,
            "solved_problems": solved_problems,
            "solve_rate": solve_rate,
            "total_cost": summary["total_cost"],
            "total_attempts": summary["total_attempts"],
            "escalations": summary["escalations"],
            "prompt_tokens": summary["prompt_tokens"],
            "completion_tokens": summary["completion_tokens"],
            "long_context_original_chars": summary["long_context_original_chars"],
            "long_context_compressed_chars": summary["long_context_compressed_chars"],
            "long_context_chars_saved": (
                summary["long_context_original_chars"]
                - summary["long_context_compressed_chars"]
            ),
            "model_usage": {row["model_id"]: row["count"] for row in model_rows},
        }
    finally:
        conn.close()
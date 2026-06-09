import sqlite3
from pathlib import Path

from fastapi import FastAPI

from server.interfaces import SolveRequest, SolveResponse

app = FastAPI()


def get_db_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    return base_dir / "metrics" / "metrics.db"


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
        conn.commit()
    finally:
        conn.close()


def log_routing_decision(solve_request: SolveRequest) -> None:
    """
    Phase 1 MVP metric logging.

    All metrics collection happens in the server. local-inference sends run_id,
    problem data, router-selected model_id, and router reasoning; the server
    records the routing metric here.
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


def get_run_optimizations(run_id: str) -> dict | None:
    """
    Returns the pre-inserted optimization metadata for run_id, if present.

    The optimizations table is intentionally populated before a run starts, so
    this row is the server-side source of truth for what each run_id means.
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


@app.post("/solve", response_model=SolveResponse)
async def solve(solve_request: SolveRequest) -> SolveResponse:
    """
    Accepts a routed problem from local-inference.

    Phase 1 establishes the server contract and centralizes metric ownership.
    The actual resolution loop will be implemented in the next phase. For now,
    the server records the routing decision and returns a shaped placeholder
    response.
    """

    log_routing_decision(solve_request)
    _run_optimizations = get_run_optimizations(solve_request.run_id)

    return SolveResponse(
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
        escalated=False,
        error="Resolution loop not implemented yet.",
    )


@app.get("/complete")
async def complete():
    """
    Returns if all tasks submitted to the server have been resolved.

    Phase 1 placeholder: /solve is synchronous for the MVP contract, so there
    are no background tasks yet.
    """

    return {"complete": True}


@app.get("/metrics")
async def metrics():
    """
    Returns basic MVP metric counts from the server-owned SQLite database.
    """

    conn = sqlite3.connect(get_db_path())
    try:
        routing_count = conn.execute("SELECT COUNT(*) FROM routing").fetchone()[0]
        problem_solving_count = conn.execute("SELECT COUNT(*) FROM problem_solving").fetchone()[0]
        optimization_runs = conn.execute("SELECT COUNT(*) FROM optimizations").fetchone()[0]
        return {
            "routing_count": routing_count,
            "problem_solving_count": problem_solving_count,
            "optimization_runs": optimization_runs,
        }
    finally:
        conn.close()
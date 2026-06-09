import sqlite3
from pathlib import Path
from .interfaces import SolveRequest
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.on_event("startup")
def on_startup():
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "metrics" / "metrics.db"
    schema_path = base_dir / "metrics" / "schema.sql"
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    try:
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    except sqlite3.OperationalError as e:
        if "already exists" not in str(e):
            raise
    finally:
        conn.close()

@app.post("/solve")
async def solve(solve_request: SolveRequest):
    """
    
    """
    pass

@app.get("/complete")
async def complete():
    """
    Returns if all tasks submitted to the server have been resolved.
    """
    pass

@app.get("/metrics")
async def metrics():
    """
    Returns metrics on which questions took up a lot of tokens/costs, which models really struggled to solve the probelms, escalations, etc.
    """
    pass


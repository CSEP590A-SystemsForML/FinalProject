from interfaces import SolveRequest
from fastapi import FastAPI

app = FastAPI()

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


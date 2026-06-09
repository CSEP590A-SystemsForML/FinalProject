from typing import Any

from pydantic import BaseModel, Field


class ProblemPayload(BaseModel):
    """
    Canonical MVP problem shape.

    Local inference owns problem loading and run_id creation, but the server owns all
    metrics collection. Each /solve request should include enough problem data for
    the server to route-log, solve, validate, and later write cost/tool metrics.
    """

    problem_id: int = Field(..., description="Stable problem id from the problem set.")
    problem: str = Field(..., description="Problem text shown to the model.")
    answer: str | None = Field(default=None, description="Expected answer or reference answer.")
    verify: str = Field(default="match", description="Validation mode: match, tests, judge, or heuristic.")
    difficulty: str = Field(default="unknown", description="Problem difficulty label.")
    category: str | None = Field(default=None, description="Optional problem category, e.g. math, code, web.")
    assert_cases: str | None = Field(default=None, description="Optional assert cases for code validation.")
    source_url: str | None = Field(default=None, description="Optional source URL for web/factual problems.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional MVP escape hatch for extra fields.")


class OptimizationFlags(BaseModel):
    """
    Optimization flags for one run.

    The optimizations table is intended to be pre-populated before a benchmark run.
    These flags may still be supplied in API responses or ad-hoc requests for
    convenience, but the server should treat the DB row for run_id as authoritative.
    """

    baseline: bool = False
    caveman: bool = False
    capabilities_prompt: bool = False
    quantized_local_lm: bool = False
    quantized_kv_cache: bool = False
    web_search_compression: bool = False
    local_model_solves: bool = False
    long_context_compression_lemma: bool = False
    long_context_compression_ai: bool = False


class SolveRequest(BaseModel):
    """
    Canonical MVP /solve request.

    local-inference creates run_id before the run and passes it here. The server logs
    routing metrics from this request and later owns all solving/cost/tool metrics.
    """

    run_id: str = Field(..., description="Pre-created benchmark run id.")
    problem_id: int = Field(..., description="Stable problem id from the problem set.")
    problem: str = Field(..., description="Problem text shown to the model.")
    answer: str | None = Field(default=None, description="Expected answer or reference answer.")
    verify: str = Field(default="match", description="Validation mode: match, tests, judge, or heuristic.")
    difficulty: str = Field(default="unknown", description="Problem difficulty label.")
    model_id: str = Field(..., description="Router-selected external model id.")
    router_reasoning: str | None = Field(default=None, description="Router reasoning for choosing model_id.")
    max_attempts: int = Field(default=2, ge=1, description="Maximum attempts before escalation/failure.")
    category: str | None = Field(default=None, description="Optional problem category, e.g. math, code, web.")
    assert_cases: str | None = Field(default=None, description="Optional assert cases for code validation.")
    source_url: str | None = Field(default=None, description="Optional source URL for web/factual problems.")
    optimizations: OptimizationFlags | None = Field(
        default=None,
        description="Optional/debug flags; server DB row for run_id is authoritative.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional MVP escape hatch for extra fields.")

    def to_problem_payload(self) -> ProblemPayload:
        return ProblemPayload(
            problem_id=self.problem_id,
            problem=self.problem,
            answer=self.answer,
            verify=self.verify,
            difficulty=self.difficulty,
            category=self.category,
            assert_cases=self.assert_cases,
            source_url=self.source_url,
            metadata=self.metadata,
        )


class LocalSolveRequest(BaseModel):
    """
    Request used when local-inference handles a very easy problem without an
    external model call.

    Metrics still go through the server so routing/problem-solving accounting
    stays centralized.
    """

    run_id: str
    problem_id: int
    problem: str
    answer: str | None = None
    verify: str = "match"
    difficulty: str = "very_easy"
    category: str | None = None
    final_answer: str
    model_id: str = "local-router"
    router_reasoning: str | None = "Solved locally by local_model_solves optimization."


class SolveResponse(BaseModel):
    """
    Canonical MVP /solve response.
    """

    run_id: str
    problem_id: int
    model_id: str
    solved: bool = False
    attempts: int = 0
    final_answer: str | None = None
    num_tool_calls: int = 0
    tool_invocations: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    escalated: bool = False
    error: str | None = None


class ModelCallResult(BaseModel):
    """
    Structured result from one external model call.
    """

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_id: str
    error: str | None = None


class CompletionConfig(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class InferenceConfig(BaseModel):
    completions: list[CompletionConfig]


class ModelConfig(BaseModel):
    source: str
    source_url: str
    id: str
    total_params: float
    active_params: float

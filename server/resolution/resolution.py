from server.cost.cost_function import calculate_model_call_cost
from server.interfaces import ModelCallResult, SolveRequest, SolveResponse
from server.utils import query_model
from server.validation import registry
from server.validation.utils import direct_match, model_judge, run_code


STRONGEST_MODEL_ID = "moonshotai/kimi-k2.6:free"


def normalize_verify_mode(verify: str | None) -> str:
    if not verify:
        return "match"

    normalized = verify.strip().lower()
    aliases = {
        "m": "match",
        "exact": "match",
        "direct": "match",
        "t": "tests",
        "test": "tests",
        "h": "heuristic",
        "heuristics": "heuristic",
        "j": "judge",
    }
    return aliases.get(normalized, normalized)


def build_solver_messages(solve_request: SolveRequest) -> list[dict[str, str]]:
    """
    Build the minimal MVP solver prompt.

    Tool instructions are intentionally omitted until tool-server integration.
    """

    verify_mode = normalize_verify_mode(solve_request.verify)

    if verify_mode == "tests":
        answer_instruction = (
            "Return only the code needed to solve the problem. Do not wrap it in markdown."
        )
    elif verify_mode == "match":
        answer_instruction = (
            "Return only the final answer. Keep it as short as possible."
        )
    else:
        answer_instruction = (
            "Return the answer clearly and concisely. Avoid unnecessary explanation."
        )

    return [
        {
            "role": "system",
            "content": (
                "You solve benchmark problems. Follow the requested answer format exactly. "
                f"{answer_instruction}"
            ),
        },
        {
            "role": "user",
            "content": solve_request.problem,
        },
    ]


def validate_answer(solve_request: SolveRequest, answer: str) -> bool:
    verify_mode = normalize_verify_mode(solve_request.verify)

    if verify_mode == "match":
        return direct_match(answer, solve_request.answer or "")

    if verify_mode == "tests":
        if not solve_request.assert_cases:
            return False
        return run_code(answer, solve_request.assert_cases)

    if verify_mode == "judge":
        return model_judge(solve_request.problem, answer, solve_request.answer or "")

    if verify_mode == "heuristic":
        return registry.test(
            solve_request.problem_id,
            True,
            answer,
            solve_request.answer or "",
        )

    return direct_match(answer, solve_request.answer or "")


def cost_model_call(call_result: ModelCallResult) -> float:
    if call_result.error:
        return 0.0

    try:
        return calculate_model_call_cost(
            call_result.model_id,
            call_result.prompt_tokens,
            call_result.completion_tokens,
        )
    except Exception:
        # Cost should not break the resolution loop in MVP.
        return 0.0


async def solve_problem(solve_request: SolveRequest) -> SolveResponse:
    """
    Minimal MVP resolution loop.

    Flow:
    - Try router-selected model up to max_attempts.
    - Validate each answer.
    - If all selected-model attempts fail, escalate once to the strongest model.
    - Return a SolveResponse with attempts, tokens, cost, and error details.
    """

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    attempts = 0
    final_answer = None
    error = None
    solved = False
    escalated = False
    final_model_id = solve_request.model_id

    messages = build_solver_messages(solve_request)

    for _ in range(solve_request.max_attempts):
        attempts += 1
        call_result = query_model(solve_request.model_id, messages)
        final_model_id = solve_request.model_id
        total_prompt_tokens += call_result.prompt_tokens
        total_completion_tokens += call_result.completion_tokens
        total_cost += cost_model_call(call_result)

        if call_result.error:
            error = call_result.error
            break

        final_answer = call_result.text
        if validate_answer(solve_request, final_answer):
            solved = True
            error = None
            break

    if not solved and solve_request.model_id != STRONGEST_MODEL_ID:
        escalated = True
        attempts += 1
        call_result = query_model(STRONGEST_MODEL_ID, messages)
        final_model_id = STRONGEST_MODEL_ID
        total_prompt_tokens += call_result.prompt_tokens
        total_completion_tokens += call_result.completion_tokens
        total_cost += cost_model_call(call_result)

        if call_result.error:
            error = call_result.error
        else:
            final_answer = call_result.text
            solved = validate_answer(solve_request, final_answer)
            if solved:
                error = None

    if not solved and error is None:
        error = "Model attempts did not validate."

    return SolveResponse(
        run_id=solve_request.run_id,
        problem_id=solve_request.problem_id,
        model_id=final_model_id,
        solved=solved,
        attempts=attempts,
        final_answer=final_answer,
        num_tool_calls=0,
        tool_invocations=[],
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_cost=total_cost,
        escalated=escalated,
        error=error,
    )
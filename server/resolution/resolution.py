from server.cost.cost_function import calculate_model_call_cost
from server.interfaces import ModelCallResult, SolveRequest, SolveResponse
from server.resolution.optimizations import caveman_prompt, compress_web_search
from server.tools import truncate_web_context, web_search_tool
from server.utils import query_model
from server.validation.utils import normalize_verify_mode, validate


STRONGEST_MODEL_ID = "moonshotai/kimi-k2.6:free"


def _optimization_enabled(run_optimizations: dict | None, key: str) -> bool:
    if not run_optimizations:
        return False
    return bool(run_optimizations.get(key))


async def prepare_tool_context(
    solve_request: SolveRequest,
    run_optimizations: dict | None = None,
) -> tuple[str, list[str], dict[str, int]]:
    """
    Deterministic MVP tool use.

    If a problem is categorized as web and has a source_url, fetch the page and
    append truncated content to the solver context. This avoids model-native tool
    calling while still measuring tool usage in the server.
    """

    tool_invocations = []
    context_parts = []
    metadata = {
        "web_context_original_chars": 0,
        "web_context_sent_chars": 0,
    }

    if solve_request.category == "web" and solve_request.source_url:
        result = await web_search_tool(solve_request.source_url)
        tool_invocations.append(result.name)
        if result.ok:
            original_web_context = result.output
            web_context = original_web_context
            if _optimization_enabled(run_optimizations, "web_search_compression"):
                web_context = compress_web_search(web_context)
            sent_web_context = truncate_web_context(web_context)
            metadata["web_context_original_chars"] += len(original_web_context)
            metadata["web_context_sent_chars"] += len(sent_web_context)
            context_parts.append(
                "Fetched web context:\n"
                f"{sent_web_context}"
            )
        else:
            context_parts.append(
                "Web fetch failed:\n"
                f"{result.error or 'unknown error'}"
            )

    return "\n\n".join(context_parts), tool_invocations, metadata


def build_solver_messages(
    solve_request: SolveRequest,
    tool_context: str = "",
    run_optimizations: dict | None = None,
) -> list[dict[str, str]]:
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

    system_prompt = (
        "You solve benchmark problems. Follow the requested answer format exactly. "
        f"{answer_instruction}"
    )
    if _optimization_enabled(run_optimizations, "caveman"):
        system_prompt = caveman_prompt(system_prompt)

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"{solve_request.problem}\n\n{tool_context}"
                if tool_context
                else solve_request.problem
            ),
        },
    ]


def validate_answer(solve_request: SolveRequest, answer: str) -> bool:
    return validate(
        problem=solve_request.problem,
        model_answer=answer,
        expected_answer=solve_request.answer,
        verify=solve_request.verify,
        problem_id=solve_request.problem_id,
        assert_cases=solve_request.assert_cases,
    )


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


async def solve_problem(
    solve_request: SolveRequest,
    run_optimizations: dict | None = None,
) -> SolveResponse:
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

    tool_context, tool_invocations, tool_metadata = await prepare_tool_context(solve_request, run_optimizations)
    messages = build_solver_messages(solve_request, tool_context, run_optimizations)

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
        num_tool_calls=len(tool_invocations),
        tool_invocations=tool_invocations,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_cost=total_cost,
        web_context_original_chars=tool_metadata["web_context_original_chars"],
        web_context_sent_chars=tool_metadata["web_context_sent_chars"],
        escalated=escalated,
        error=error,
    )
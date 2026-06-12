import json
import re

from server.cost.accumulator import CostAccumulator
from server.interfaces import SolveRequest, SolveResponse
from server.resolution.ladder import model_ladder, resolve_start_index
from server.resolution.optimizations import (
    caveman_prompt,
    compress_web_search,
    long_context_compression_ai,
    long_context_compression_lemma,
)
from server.tools import run_candidate_code_tool, truncate_web_context, web_search_tool
from server.utils import query_model
from server.validation.utils import normalize_verify_mode, validate


# The escalation ladder (weakest -> strongest) is derived from configs/models.yaml
# by server.resolution.ladder. STRONGEST_MODEL_ID is kept as the top rung for any
# callers/tests that still import it.
STRONGEST_MODEL_ID = model_ladder()[-1]

# Confidence below this raises the starting rung by one (see ladder.resolve_start_index).
LOW_CONFIDENCE_THRESHOLD = 0.45

# When there is no ground truth (general/specialized use: only a prompt), a rung's
# answer is "accepted" only if the solver self-reports at least this confidence.
# Otherwise we give up on that rung and escalate to the next, larger model.
ACCEPT_CONFIDENCE_THRESHOLD = 0.60


def _optimization_enabled(run_optimizations: dict | None, key: str) -> bool:
    if not run_optimizations:
        return False
    return bool(run_optimizations.get(key))


def _apply_long_context_compression(
    text: str,
    run_optimizations: dict | None,
) -> str:
    """
    Shrink an over-long solver context, gated on the run's flags.

    `_ai` takes precedence over `_lemma` when both are set. Both functions no-op
    below their char threshold, so short prompts pass through untouched.
    """

    if _optimization_enabled(run_optimizations, "long_context_compression_ai"):
        return long_context_compression_ai(text)
    if _optimization_enabled(run_optimizations, "long_context_compression_lemma"):
        return long_context_compression_lemma(text)
    return text


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
    request_confidence: bool = False,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """
    Build the minimal MVP solver prompt.

    Tool use is deterministic and server-driven (web context is fetched in
    prepare_tool_context; code is run + repaired in the solve loop), so the
    prompt deliberately omits model-native function-calling instructions.

    When `request_confidence` is True (no ground truth is available to grade the
    answer), the solver is asked to return a JSON object with both its answer and
    a calibrated self-confidence, which the escalation loop uses as the accept /
    give-up signal in place of a validator.

    Returns the chat messages plus long-context-compression metadata
    (original vs. compressed user-content chars) so the savings can be recorded
    and quoted. When compression is disabled / under threshold the two char
    counts are equal (i.e. zero savings).
    """

    verify_mode = normalize_verify_mode(solve_request.verify)

    if request_confidence:
        answer_instruction = (
            "There is no answer key. Solve the problem, then return STRICT JSON only, "
            'in the form {"answer": "<your final answer or code>", "confidence": <float 0..1>}. '
            "confidence is your calibrated probability that the answer is correct. "
            "Be honest: use a low confidence when unsure. Do not wrap the JSON in markdown."
        )
    elif verify_mode == "tests":
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

    user_content = (
        f"{solve_request.problem}\n\n{tool_context}"
        if tool_context
        else solve_request.problem
    )
    original_chars = len(user_content)
    # Code specs can be whitespace-sensitive, so leave `tests` prompts intact.
    if verify_mode != "tests":
        user_content = _apply_long_context_compression(user_content, run_optimizations)
    long_context_metadata = {
        "long_context_original_chars": original_chars,
        "long_context_compressed_chars": len(user_content),
    }

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages, long_context_metadata


MAX_REPAIR_FEEDBACK_CHARS = 2_000


def _code_repair_messages(candidate_code: str, exec_result) -> list[dict[str, str]]:
    """
    Build the assistant+user turn that shows the solver its failing test output
    so it can repair the code on the next attempt.
    """

    feedback = (exec_result.output or exec_result.error or "tests failed").strip()
    if len(feedback) > MAX_REPAIR_FEEDBACK_CHARS:
        feedback = feedback[:MAX_REPAIR_FEEDBACK_CHARS] + " [truncated]"

    return [
        {"role": "assistant", "content": candidate_code},
        {
            "role": "user",
            "content": (
                "Your code did not pass its tests. Runner output:\n"
                f"{feedback}\n\n"
                "Return only the corrected code. Do not wrap it in markdown."
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
        validator=solve_request.validator,
    )


def has_ground_truth(solve_request: SolveRequest) -> bool:
    """
    Whether the problem ships something to grade an answer against.

    In general/specialized use the caller has only a prompt (no answer, no tests,
    no named validator). Then `validate_answer` cannot be trusted, and the
    escalation loop falls back to the solver's self-reported confidence instead.
    """

    verify_mode = normalize_verify_mode(solve_request.verify)
    if verify_mode == "tests":
        return bool(solve_request.assert_cases)
    if verify_mode in {"match", "judge"}:
        return bool(solve_request.answer)
    if verify_mode == "heuristic":
        return bool(solve_request.validator) or bool(solve_request.answer)
    return False


_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')


def parse_answer_confidence(text: str) -> tuple[str, float | None]:
    """
    Pull (answer, confidence) out of a no-ground-truth solver response.

    The solver is asked for strict JSON {"answer", "confidence"}, but MVP models
    wrap or malform it, so parse defensively: try JSON first, then a regex for the
    confidence, and finally fall back to the raw text with unknown confidence
    (which the loop treats as "not confident" -> escalate).
    """

    if not text:
        return "", None

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        blob = match.group(0)
        for candidate in (blob, blob.replace("'", '"')):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                answer = parsed.get("answer", text)
                conf = parsed.get("confidence")
                try:
                    conf = None if conf is None else max(0.0, min(1.0, float(conf)))
                except (TypeError, ValueError):
                    conf = None
                return (str(answer), conf)

    conf_match = _CONFIDENCE_RE.search(text)
    if conf_match:
        try:
            return text, max(0.0, min(1.0, float(conf_match.group(1))))
        except ValueError:
            pass
    return text, None


DEFAULT_ROUTING_STRATEGY = "confidence"

_STRATEGY_ALIASES = {
    "confidence": "confidence",
    "ladder": "confidence",
    "difficulty": "difficulty",
    "legacy": "difficulty",
    "baseline": "difficulty",
}


def normalize_routing_strategy(
    strategy: str | None,
    run_optimizations: dict | None = None,
) -> str:
    """
    Resolve which resolution strategy to run.

    Precedence: explicit SolveRequest.routing_strategy, then a run-level
    `routing_strategy` optimization flag, then the server default. Unknown values
    fall back to the default so a typo never silently disables routing.
    """

    for candidate in (strategy, (run_optimizations or {}).get("routing_strategy")):
        if candidate:
            resolved = _STRATEGY_ALIASES.get(str(candidate).strip().lower())
            if resolved:
                return resolved
    return DEFAULT_ROUTING_STRATEGY


def _build_response(
    solve_request: SolveRequest,
    *,
    model_id: str,
    solved: bool,
    attempts: int,
    final_answer: str | None,
    cost: CostAccumulator,
    tool_invocations: list[str],
    tool_metadata: dict[str, int],
    long_context_metadata: dict[str, int],
    escalated: bool,
    error: str | None,
) -> SolveResponse:
    return SolveResponse(
        run_id=solve_request.run_id,
        problem_id=solve_request.problem_id,
        model_id=model_id,
        solved=solved,
        attempts=attempts,
        final_answer=final_answer,
        num_tool_calls=len(tool_invocations),
        tool_invocations=tool_invocations,
        prompt_tokens=cost.prompt_tokens,
        completion_tokens=cost.completion_tokens,
        total_cost=cost.cost,
        web_context_original_chars=tool_metadata["web_context_original_chars"],
        web_context_sent_chars=tool_metadata["web_context_sent_chars"],
        long_context_original_chars=long_context_metadata["long_context_original_chars"],
        long_context_compressed_chars=long_context_metadata["long_context_compressed_chars"],
        escalated=escalated,
        error=error,
    )


async def solve_problem(
    solve_request: SolveRequest,
    run_optimizations: dict | None = None,
) -> SolveResponse:
    """
    Resolve one problem, dispatching to the requested routing strategy.

    Both strategies share metrics, tool use, and the cost accumulator so their
    results are directly comparable across a benchmark:

    - 'confidence' (default): confidence-routed start rung + gradual ladder
      escalation + ground-truth-optional acceptance. See `_solve_confidence`.
    - 'difficulty' (legacy): use the router's difficulty-based model pick, then
      escalate once to the strongest model on failure. See `_solve_difficulty`.
    """

    strategy = normalize_routing_strategy(solve_request.routing_strategy, run_optimizations)
    if strategy == "difficulty":
        return await _solve_difficulty(solve_request, run_optimizations)
    return await _solve_confidence(solve_request, run_optimizations)


async def _solve_confidence(
    solve_request: SolveRequest,
    run_optimizations: dict | None = None,
) -> SolveResponse:
    """
    Confidence-routed, ladder-escalating resolution loop.

    Flow:
    - Pick a STARTING rung on the capability ladder from prompt-only signals: the
      router's model pick, its inferred difficulty, and its confidence
      (low confidence -> start one rung higher).
    - At each rung, try the model up to max_attempts (with solve->run->repair for
      code), then decide accept vs. give-up:
        * ground truth available -> accept iff the answer validates.
        * no ground truth        -> accept iff the solver self-reports confidence
                                     >= ACCEPT_CONFIDENCE_THRESHOLD.
    - On give-up (or a hard call error such as a rate limit), ESCALATE to the next
      larger model on the ladder. Stop on accept or when the ladder is exhausted.
    """

    cost = CostAccumulator()
    ladder = model_ladder()
    grounded = has_ground_truth(solve_request)

    verify_mode = normalize_verify_mode(solve_request.verify)
    runs_code_tests = verify_mode == "tests" and bool(solve_request.assert_cases)

    start_index = resolve_start_index(
        model_id=solve_request.model_id,
        difficulty=solve_request.difficulty_pred,
        confidence=solve_request.confidence,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
    )

    tool_context, tool_invocations, tool_metadata = await prepare_tool_context(solve_request, run_optimizations)
    base_messages, long_context_metadata = build_solver_messages(
        solve_request, tool_context, run_optimizations, request_confidence=not grounded
    )

    attempts = 0
    final_answer = None
    error = None
    solved = False
    final_model_id = ladder[start_index] if ladder else solve_request.model_id

    for rung_index in range(start_index, len(ladder)):
        model_id = ladder[rung_index]
        final_model_id = model_id
        messages = list(base_messages)
        rung_accepted = False

        for attempt_index in range(solve_request.max_attempts):
            attempts += 1
            call_result = query_model(model_id, messages)
            cost.add(call_result)

            if call_result.error:
                # Hard failure (e.g. rate limit): give up on this rung, escalate.
                error = call_result.error
                break

            if grounded:
                final_answer = call_result.text
                if validate_answer(solve_request, final_answer):
                    rung_accepted = True
                    error = None
                    break
                # solve -> run -> repair: for code problems, run the candidate
                # against its asserts and feed the failure back so the next attempt
                # (or the next rung) can fix it.
                if runs_code_tests and attempt_index < solve_request.max_attempts - 1:
                    exec_result = await run_candidate_code_tool(final_answer, solve_request.assert_cases)
                    tool_invocations.append(exec_result.name)
                    messages = messages + _code_repair_messages(final_answer, exec_result)
            else:
                final_answer, self_conf = parse_answer_confidence(call_result.text)
                if self_conf is not None and self_conf >= ACCEPT_CONFIDENCE_THRESHOLD:
                    rung_accepted = True
                    error = None
                    break
                # Not confident enough; no validator to repair against, so stop
                # retrying this rung and escalate to a stronger model.
                error = "Solver confidence below acceptance threshold."
                break

        if rung_accepted:
            solved = True
            break
        # otherwise: give up on this (smaller) model and escalate up the ladder.

    escalated = final_model_id != ladder[start_index] if ladder else False

    if not solved and error is None:
        error = "Model attempts did not validate."

    return _build_response(
        solve_request,
        model_id=final_model_id,
        solved=solved,
        attempts=attempts,
        final_answer=final_answer,
        cost=cost,
        tool_invocations=tool_invocations,
        tool_metadata=tool_metadata,
        long_context_metadata=long_context_metadata,
        escalated=escalated,
        error=error,
    )


async def _solve_difficulty(
    solve_request: SolveRequest,
    run_optimizations: dict | None = None,
) -> SolveResponse:
    """
    Legacy difficulty-routed resolution loop (the earlier path, kept for A/B).

    The router has already picked a model from the problem's inferred difficulty;
    here the server simply trusts that pick, tries it up to max_attempts (with
    solve->run->repair for code), and escalates ONCE to the strongest model if
    every attempt fails to validate. No capability ladder, no confidence signal.
    """

    cost = CostAccumulator()
    attempts = 0
    final_answer = None
    error = None
    solved = False
    escalated = False
    final_model_id = solve_request.model_id

    verify_mode = normalize_verify_mode(solve_request.verify)
    runs_code_tests = verify_mode == "tests" and bool(solve_request.assert_cases)

    tool_context, tool_invocations, tool_metadata = await prepare_tool_context(solve_request, run_optimizations)
    messages, long_context_metadata = build_solver_messages(solve_request, tool_context, run_optimizations)

    for attempt_index in range(solve_request.max_attempts):
        attempts += 1
        call_result = query_model(solve_request.model_id, messages)
        final_model_id = solve_request.model_id
        cost.add(call_result)

        if call_result.error:
            error = call_result.error
            break

        final_answer = call_result.text
        if validate_answer(solve_request, final_answer):
            solved = True
            error = None
            break

        if runs_code_tests and attempt_index < solve_request.max_attempts - 1:
            exec_result = await run_candidate_code_tool(final_answer, solve_request.assert_cases)
            tool_invocations.append(exec_result.name)
            messages = messages + _code_repair_messages(final_answer, exec_result)

    if not solved and solve_request.model_id != STRONGEST_MODEL_ID:
        escalated = True
        attempts += 1
        call_result = query_model(STRONGEST_MODEL_ID, messages)
        final_model_id = STRONGEST_MODEL_ID
        cost.add(call_result)

        if call_result.error:
            error = call_result.error
        else:
            final_answer = call_result.text
            solved = validate_answer(solve_request, final_answer)
            if solved:
                error = None

    if not solved and error is None:
        error = "Model attempts did not validate."

    return _build_response(
        solve_request,
        model_id=final_model_id,
        solved=solved,
        attempts=attempts,
        final_answer=final_answer,
        cost=cost,
        tool_invocations=tool_invocations,
        tool_metadata=tool_metadata,
        long_context_metadata=long_context_metadata,
        escalated=escalated,
        error=error,
    )
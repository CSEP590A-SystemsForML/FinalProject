def run_code(function: str, assert_cases: str) -> bool:
    """
    Executes the given Python function definition combined with assert_cases.
    Returns True if execution succeeds without raising any exceptions, and False otherwise.
    """
    if not function:
        return False
    try:
        # Execute the code within an isolated dictionary environment
        env = {}
        exec(f"{function}\n{assert_cases}", env)
        return True
    except Exception:
        return False


def direct_match(model_answer: str, correct_answer: str) -> bool:
    """
    Compares the model's answer directly with the correct answer.
    Returns True if they are identical after stripping leading/trailing whitespace.
    """
    if model_answer is None or correct_answer is None:
        return False
    return model_answer.strip() == correct_answer.strip()


def model_judge(question: str, model_answer: str, correct_answer: str) -> bool:
    """
    Uses an LLM judge to determine if the model's answer is correct relative to the reference correct answer.
    """
    if not question or not model_answer or not correct_answer:
        return False

    from server.utils import query_model

    prompt = (
        f"You are a grading assistant.\n"
        f"Evaluate whether the student's answer is correct based on the question and the reference correct answer.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Student Answer: {model_answer}\n\n"
        f"Is the Student Answer correct? Reply with exactly 'yes' if it is correct, or 'no' if it is incorrect. Do not add any other words or explanation."
    )

    model_id = "openai/gpt-oss-120b:free"
    response = query_model(model_id, prompt)

    if response.error or not response.text:
        return False

    res_clean = response.text.strip().lower().strip(" .!,;:")
    return "yes" in res_clean or "true" in res_clean
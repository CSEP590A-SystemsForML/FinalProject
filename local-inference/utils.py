import asyncio
import json
import os
import re
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
import yaml
from openai import AsyncOpenAI


VALID_VERIFY_MODES = {"match", "tests", "judge", "heuristic"}
VALID_DIFFICULTIES = {"very_easy", "easy", "medium", "hard", "very_hard"}


def validate_problem_set(problem_set: pd.DataFrame) -> None:
    """
    Validate the loaded problem set and fail loudly on the FIRST malformed run,
    listing every offending problem at once.

    Schema (see PROGRESS.md "Locked problem schema"):
    - problem_id: unique integer
    - problem: non-empty text
    - verify: one of match | tests | judge | heuristic
    - difficulty: one of very_easy | easy | medium | hard | very_hard
    - category: non-empty string
    - answer: required when verify is match or judge
    - assert_cases: required when verify is tests
    - validator: required when verify is heuristic
    """

    errors: list[str] = []

    required_columns = {"problem_id", "problem", "verify", "difficulty", "category"}
    missing_columns = required_columns - set(problem_set.columns)
    if missing_columns:
        raise ValueError(
            f"Problem set is missing required columns: {sorted(missing_columns)}"
        )

    seen_ids: dict[object, int] = {}

    for position, row in problem_set.iterrows():
        record = row.to_dict()
        pid = record.get("problem_id")
        label = f"problem_id={pid}" if pd.notna(pid) else f"row {position}"

        if pd.isna(pid):
            errors.append(f"{label}: problem_id is missing.")
        else:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                errors.append(f"{label}: problem_id must be an integer.")
            else:
                if pid_int in seen_ids:
                    errors.append(f"{label}: duplicate problem_id (also row {seen_ids[pid_int]}).")
                seen_ids[pid_int] = position

        problem_text = record.get("problem")
        if not isinstance(problem_text, str) or not problem_text.strip():
            errors.append(f"{label}: 'problem' must be non-empty text.")

        verify = str(record.get("verify", "")).strip().lower()
        if verify not in VALID_VERIFY_MODES:
            errors.append(
                f"{label}: verify='{record.get('verify')}' invalid; expected one of {sorted(VALID_VERIFY_MODES)}."
            )

        difficulty = str(record.get("difficulty", "")).strip().lower()
        if difficulty not in VALID_DIFFICULTIES:
            errors.append(
                f"{label}: difficulty='{record.get('difficulty')}' invalid; expected one of {sorted(VALID_DIFFICULTIES)}."
            )

        category = record.get("category")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"{label}: 'category' must be a non-empty string.")

        def _has(field: str) -> bool:
            value = record.get(field)
            return value is not None and not (pd.isna(value) if not isinstance(value, str) else not value.strip())

        if verify in {"match", "judge"} and not _has("answer"):
            errors.append(f"{label}: verify='{verify}' requires a non-empty 'answer'.")
        if verify == "tests" and not _has("assert_cases"):
            errors.append(f"{label}: verify='tests' requires non-empty 'assert_cases'.")
        if verify == "heuristic" and not _has("validator"):
            errors.append(
                f"{label}: verify='heuristic' requires a 'validator' name registered in server/validation/registry.py."
            )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            f"Problem set failed validation ({len(errors)} issue(s)):\n  - {joined}"
        )


def _domains_dir() -> Path:
    return Path(__file__).resolve().parent / "problems" / "domains"


def available_domains() -> list[str]:
    """Domain names with a built dataset file, sorted for deterministic loading."""
    return sorted(p.stem for p in _domains_dir().glob("*.json"))


def load_domain_problems(domain: str | None) -> pd.DataFrame:
    """
    Load problems for one domain, or all domains concatenated when domain is None.

    Domain datasets are built by problems/build.py into problems/domains/<domain>.json.
    """
    domains_dir = _domains_dir()
    all_domains = available_domains()
    if not all_domains:
        raise FileNotFoundError(
            f"No domain datasets found in {domains_dir}. Run: python {domains_dir.parent}/build.py"
        )

    if domain is None:
        selected = all_domains
    else:
        if domain not in all_domains:
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {all_domains}"
            )
        selected = [domain]

    frames = [pd.read_json(domains_dir / f"{name}.json") for name in selected]
    return pd.concat(frames, ignore_index=True)


class ProblemSetManager:
    """
    Class for managing the dataset of problems to solve and handing off the next problem when requested.

    Pass `domain` to scope the run to a single domain (e.g. "code", "math"); leave
    it None to run across all domains.
    """

    def __init__(
        self,
        limit: int | None = None,
        problem_id: int | None = None,
        domain: str | None = None,
    ) -> None:
        self.problem_set = load_domain_problems(domain)
        if "problem_id" not in self.problem_set.columns:
            self.problem_set["problem_id"] = range(len(self.problem_set))

        validate_problem_set(self.problem_set)

        if problem_id is not None:
            self.problem_set = self.problem_set[self.problem_set["problem_id"] == problem_id]
            if self.problem_set.empty:
                raise ValueError(f"No problem found with problem_id={problem_id}")

        self.problem_set = self.problem_set.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be a positive integer")
            self.problem_set = self.problem_set.head(limit).reset_index(drop=True)

        self.index = 0

    def get_next_problem(self):
        if self.index >= len(self.problem_set):
            return None
        problem = self.problem_set.iloc[self.index]
        self.index += 1
        return problem.to_dict()


class LocalInferenceManager:
    def __init__(
        self,
        max_active: int,
        router_base_url: str | None = None,
        router_model: str | None = None,
    ) -> None:
        self.router_client = AsyncOpenAI(
            api_key="dummy",
            base_url=router_base_url or os.environ.get("ROUTER_BASE_URL", "http://localhost:7654/v1"),
        )
        self.model = router_model or os.environ.get("ROUTER_MODEL", "ibm-granite/granite-4.1-3b")
        self.active_sem = asyncio.Semaphore(max_active)

    def _extract_json_object(self, content: str) -> str:
        """
        Extract the first JSON-looking object from a model response.

        Router prompts should emit strict JSON, but MVP local models may wrap it in
        reasoning or markdown. This keeps parsing tolerant without making local
        inference responsible for metrics.
        """

        if not content:
            raise ValueError("Router returned empty content.")

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Router response did not contain a JSON object: {content}")

        return match.group(0)

    def parse_router_response(self, content: str) -> dict:
        json_text = self._extract_json_object(content)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            # Some local models follow the prompt example with single quotes.
            parsed = json.loads(json_text.replace("'", '"'))

        if not isinstance(parsed, dict):
            raise ValueError(f"Router JSON was not an object: {parsed}")

        return parsed

    async def call_model(self, messages: list[dict[str, str]]):
        async with self.active_sem:
            response = await self.router_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_completion_tokens=8192,
                stop=[
                    "<|endoftext|>",
                ],
            )
            first_res = response.choices[0].message.content
            first_parsed = self.parse_router_response(first_res)
            messages.append({"role": "assistant", "content": json.dumps(first_parsed)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Look at your own logic. Do you believe a student of the level as described in the model description is capable of this? "
                        "Emit your final response in the same format and revise your reasoning and model_id if you believe you have made a mistake."
                    ),
                }
            )
            second_res = await self.router_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_completion_tokens=8192,
                stop=[
                    "<|endoftext|>",
                ],
            )
            second_parsed = self.parse_router_response(second_res.choices[0].message.content)
        return first_parsed, second_parsed


class ResolutionServerClient:
    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")

    async def solve(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.server_url}/solve", json=payload)
            response.raise_for_status()
            return response.json()

    async def local_solve(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.server_url}/local-solve", json=payload)
            response.raise_for_status()
            return response.json()


class LogicManager:
    """
    Class for managing which requests are active and implements the core routing logic.
    Ensures that the vllm server always has adequate work and sends routed problems to the model server.

    All metric collection happens in the server. local-inference only sends the run_id,
    problem payload, router-selected model_id, and router reasoning.
    """

    def __init__(
        self,
        prompt_type: Literal["cache", "capabilities"],
        max_active: int,
        server_url: str,
        run_id: str,
        max_attempts: int,
        local_model_solves: bool = False,
    ) -> None:
        self.inference_manager = LocalInferenceManager(max_active)
        self.resolution_server = ResolutionServerClient(server_url)
        self.run_id = run_id
        self.max_attempts = max_attempts
        self.local_model_solves = local_model_solves
        self.models_config = {}
        self._init_routing_prompt(prompt_type)

    def _init_routing_prompt(self, prompt_type: Literal["cache", "capabilities"]):
        configs_path = Path(__file__).resolve().parent.parent / "configs"
        with open(configs_path / "models.yaml") as c:
            models_config = yaml.safe_load(c)
        with open(configs_path / "prompts.yaml") as c:
            prompts_config = yaml.safe_load(c)

        self.models_config = models_config
        self.valid_model_ids = set(models_config.keys())

        if prompt_type == "cache":
            models_prompt = "\n".join(
                [
                    f"{model_id}: {models_config[model_id]['total_params']}B total, {models_config[model_id]['active_params']}B active. \nQuick Description: {models_config[model_id]['description']}"
                    for model_id in models_config.keys()
                ]
            )
            self.router_prompt = (
                f"{prompts_config['router']['cache']}"
                "Your options are:\n"
                f"{models_prompt}"
            )
        else:
            with open(configs_path / "benchmarks.yaml") as c:
                benchmarks_config = yaml.safe_load(c)
            benchmarks_prompts = {
                model_id: "\n".join(
                    [
                        f"{benchmark['description']}: {benchmark['score']}"
                        for benchmark in benchmarks_config[model_id].values()
                    ]
                )
                for model_id in benchmarks_config.keys()
            }
            models_prompt = (
                "\n".join(
                    [
                        (
                            f"{model_id}: {models_config[model_id]['total_params']}B total, {models_config[model_id]['active_params']}B active.\n"
                            f"{benchmarks_prompts[model_id]}"
                        )
                        for model_id in models_config.keys()
                    ]
                )
            )
            self.router_prompt = (
                f"{prompts_config['router']['cache']}"
                f"{prompts_config['router']['capabilities']}"
                "Your options are:\n"
                f"{models_prompt}"
            )

    async def handle(self, request_type: Literal["problem"], request):
        if request_type != "problem":
            raise ValueError(
                f"Unsupported request_type={request_type!r}; only 'problem' is handled."
            )
        return await self.handle_problem(request)

    def _normalize_model_id(self, model_id: str | None) -> str:
        if model_id in self.valid_model_ids:
            return model_id

        if model_id:
            stripped = model_id.strip()
            if stripped in self.valid_model_ids:
                return stripped

        fallback_model = "moonshotai/kimi-k2.6:free"
        if fallback_model in self.valid_model_ids:
            return fallback_model

        return next(iter(self.valid_model_ids))

    def _build_solve_payload(self, request: dict, router_response: dict) -> dict:
        model_id = self._normalize_model_id(router_response.get("model_id"))
        reasoning = router_response.get("reasoning")

        return {
            "run_id": self.run_id,
            "problem_id": int(request["problem_id"]),
            "problem": str(request["problem"]),
            "answer": None if pd.isna(request.get("answer")) else str(request.get("answer")),
            "verify": str(request.get("verify", "match")),
            "difficulty": str(request.get("difficulty", "unknown")),
            "category": None if pd.isna(request.get("category")) else request.get("category"),
            "assert_cases": None if pd.isna(request.get("assert_cases")) else request.get("assert_cases"),
            "source_url": None if pd.isna(request.get("source_url")) else request.get("source_url"),
            "image_url": None if pd.isna(request.get("image_url")) else request.get("image_url"),
            "validator": None if pd.isna(request.get("validator")) else request.get("validator"),
            "model_id": model_id,
            "router_reasoning": None if reasoning is None else str(reasoning),
            "max_attempts": self.max_attempts,
        }

    def _can_solve_locally(self, request: dict) -> bool:
        return (
            self.local_model_solves
            and str(request.get("difficulty", "")).strip().lower() == "very_easy"
            and str(request.get("verify", "")).strip().lower() == "match"
            and not pd.isna(request.get("answer"))
        )

    def _build_local_solve_payload(self, request: dict) -> dict:
        return {
            "run_id": self.run_id,
            "problem_id": int(request["problem_id"]),
            "problem": str(request["problem"]),
            "answer": str(request.get("answer")),
            "verify": str(request.get("verify", "match")),
            "difficulty": str(request.get("difficulty", "very_easy")),
            "category": None if pd.isna(request.get("category")) else request.get("category"),
            "final_answer": str(request.get("answer")),
            "model_id": "local-router",
            "router_reasoning": "local_model_solves optimization handled very_easy match problem locally.",
        }

    async def handle_problem(self, request):
        if self._can_solve_locally(request):
            local_payload = self._build_local_solve_payload(request)
            local_response = await self.resolution_server.local_solve(local_payload)
            print(
                f"For problem_id={request['problem_id']}: {request['problem']}\n"
                f"Local solve response: {local_response}\n"
            )
            return local_response

        prompt = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": str(request["problem"])},
        ]
        first_res, second_res = await self.inference_manager.call_model(prompt)
        solve_payload = self._build_solve_payload(request, second_res)
        solve_response = await self.resolution_server.solve(solve_payload)

        print(
            f"For problem_id={request['problem_id']}: {request['problem']}\n"
            f"First routing response: {first_res}\n"
            f"Revised routing response: {second_res}\n"
            f"Server response: {solve_response}\n"
        )
        return solve_response

    def simple_routing_logger(self, request, model_id, reasoning):
        print("=" * 30)
        print(f"For a problem of {request['difficulty']}, router chose {model_id}")
        print("+" * 30)
        print(f"Reasoning trace: {reasoning}")
        print("-" * 30)
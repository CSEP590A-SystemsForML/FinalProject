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


class ProblemSetManager:
    """
    Class for managing the dataset of problems to solve and handing off the next problem when requested.
    """

    def __init__(self) -> None:
        self.problem_set = pd.read_json(
            Path(__file__).resolve().parent / "problems" / "problems.json"
        )
        if "problem_id" not in self.problem_set.columns:
            self.problem_set["problem_id"] = range(len(self.problem_set))
        self.problem_set = self.problem_set.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)
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
    ) -> None:
        self.inference_manager = LocalInferenceManager(max_active)
        self.resolution_server = ResolutionServerClient(server_url)
        self.run_id = run_id
        self.max_attempts = max_attempts
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

    async def handle(self, request_type: Literal["problem", "solution"], request):
        if request_type == "problem":
            return await self.handle_problem(request)
        else:
            return await self.handle_solution(request)

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
            "model_id": model_id,
            "router_reasoning": None if reasoning is None else str(reasoning),
            "max_attempts": self.max_attempts,
        }

    async def handle_problem(self, request):
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

    async def handle_solution(self, request):
        raise NotImplementedError("Local model cannot handle solutions yet.")

    def simple_routing_logger(self, request, model_id, reasoning):
        print("=" * 30)
        print(f"For a problem of {request['difficulty']}, router chose {model_id}")
        print("+" * 30)
        print(f"Reasoning trace: {reasoning}")
        print("-" * 30)
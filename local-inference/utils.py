from openai import AsyncOpenAI
from typing import Literal
import yaml
from pathlib import Path
import asyncio
import pandas as pd
import re

class ProblemSetManager:
    """
    Class for managing the dataset of problems to solve and handing off the next problem when requested..
    """
    def __init__(self) -> None:
        self.problem_set = pd.read_json(
            Path(__file__).resolve().parent / "problems" / "problems.json"
        )
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
    ) -> None:
        self.router_client = AsyncOpenAI(api_key="dummy", base_url="http://localhost:7654/v1")
        self.model = "ibm-granite/granite-4.1-3b"
        self.active_sem = asyncio.Semaphore(max_active)

    async def call_model(self, messages: list[dict[str, str]]):
        async with self.active_sem:
            response = await self.router_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_completion_tokens=8192,
                stop=[
                    "<|endoftext|>",
                ]
            )

        content = response.choices[0].message.content
        return content


class LogicManager:
    """
    Class for managing which requests are active and implements the core routing logic.
    Ensures that the vllm server always has adequate work, sends problems to the model server, and logs routing metrics.
    """
    def __init__(self,
        prompt_type: Literal["cache", "capabilities"], 
        max_active: int
    ) -> None:
        self.inference_manager = LocalInferenceManager(max_active)
        self._init_routing_prompt(prompt_type)
        
    def _init_routing_prompt(self, prompt_type: Literal["cache", "capabilities"]):
        configs_path = Path(__file__).resolve().parent.parent / "configs"
        with open(configs_path / "models.yaml") as c:
            models_config = yaml.safe_load(c)
        with open(configs_path / "prompts.yaml") as c:
            prompts_config = yaml.safe_load(c)
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
                        ) for model_id in models_config.keys()
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

    async def handle_problem(self, request):
        prompt = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": str(request["problem"])},
        ]
        content = await self.inference_manager.call_model(prompt)
        print(f"For problem: {request['problem']}\n{content}")
        #self.simple_routing_logger(request, model_id, reasoning)
        # TODO: Add validation logic here to fuzzy match to a model to be safe?.
        # TODO: Add calling fastapi server with request data and model_id
        # TODO: Add actual heurisitc logging.
        return

    async def handle_solution(self, request):
        raise NotImplemented("Local model cannot handle solutions yet.")

    def simple_routing_logger(self, request, model_id, reasoning):
        print("=" * 30)
        print(f"For a problem of {request['difficulty']}, router chose {model_id}")
        print("+" * 30)
        print(f"Reasoning trace: {reasoning}")
        print("-" * 30)
    
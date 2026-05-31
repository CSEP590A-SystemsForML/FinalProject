from openai import AsyncOpenAI
from typing import Literal
import yaml
from pathlib import Path
import asyncio
import pandas as pd

class LocalInferenceManager:
    def __init__(
        self, 
        prompt_type: Literal["cache", "capabilities"], 
        max_active: int, 
        max_queued: int
    ) -> None:
        self.router_client = AsyncOpenAI(api_key="dummy", base_url="http://localhost:8000/v1")
        self.model = "Qwen/Qwen3.5-4B"
        self.queued_sem = asyncio.Semaphore(max_queued)
        self.active_sem = asyncio.Semaphore(max_active)
        self._init_base_prompt(prompt_type)
        self._init_problem_set()

    def _init_base_prompt(self, prompt_type: Literal["cache", "capabilities"]):
        configs_path = Path(__file__).resolve().parent / "configs"
        with open(configs_path / "models.yaml") as c:
            models_config = yaml.safe_load(c)
        with open(configs_path / "prompts.yaml") as c:
            prompts_config = yaml.safe_load(c)
        if prompt_type == "cache":
            models_prompt = "\n".join(
                [
                    f"{model_id}: {models_config[model_id]['total_params']}B total, {models_config[model_id]['active_params']}B active." 
                    for model_id in models_config.keys()
                ]
            )
            self.router_prompt = (
                f"{prompts_config["router"]["cache"]}"
                "Your options are:\n"
                f"{models_prompt}"
            )
        else:
            with open(configs_path / "benchmarks.yaml") as c:
                benchmarks_config = yaml.safe_load(c)
            benchmarks_prompts = {
                model_id: "\n".join([f"{benchmark['description']}: {benchmark['score']}" for benchmark in benchmarks_config[model_id].keys()])
                for model_id in benchmarks_config.keys()
                
            }
            models_prompt = (
                "\n".join(
                    [
                        (
                            f"{model_id}: {models_config[model_id]['total_params']}B total, {models_config[model_id]['active_params']}B active."
                            f"{benchmarks_prompts[model_id]}"
                        ) for model_id in models_config.keys()
                    ]
                )
            )
            self.router_prompt = (
                f"{prompts_config["router"]["cache"]}"
                f"{prompts_config["router"]["capabilities"]}"
                "Your options are:\n"
                f"{models_prompt}"
            )

    def _init_problem_set(self):
        self.problem_set = pd.read_json(Path(__file__).resolve().parent / "problems" / "problems.json")
        self.problem_set.sample(frac=1, random_state=42).reset_index(drop=True)

    async def route_problem(self, problem_prompt: str):
        response = await self.router_client.chat.completion.create(
            model=self.model,
            messages = [
                {"system": self.router_prompt},
                {"user": problem_prompt}
            ]
        )

    
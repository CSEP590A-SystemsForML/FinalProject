from dataclasses import dataclass

@dataclass
class SolveRequest:
    model_id: str
    problem_level: str
    problem_id: int

@dataclass
class CompletionConfig:
    prompt_tokens: int
    completion_tokens: int

@dataclass
class InferenceConfig:
    completions: list[CompletionConfig]

@dataclass
class ModelConfig:
    source: str
    source_url: str
    id: str
    total_params: float
    active_params: float

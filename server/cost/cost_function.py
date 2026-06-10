from pathlib import Path

import yaml

from server.interfaces import CompletionConfig, InferenceConfig, ModelConfig


# Image tokens are priced above text input tokens: vision encoders + the raw
# token volume of images are, per the pitch, disproportionately expensive.
IMAGE_TOKEN_PREMIUM = 2.0

# Rough OpenAI-style estimate: a base cost plus per-512px-tile cost.
IMAGE_BASE_TOKENS = 85
IMAGE_TOKENS_PER_TILE = 170


def estimate_image_tokens(num_tiles: int = 4) -> int:
    """
    Estimate the prompt-token cost of one image.

    We don't always know an image's dimensions up front, so default to a
    medium-detail 4-tile image. Callers that know the tiling can pass num_tiles.
    """

    return IMAGE_BASE_TOKENS + max(0, int(num_tiles)) * IMAGE_TOKENS_PER_TILE


def calculate_cost(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    attention_fraction: float = 0.17,
    base_total_params: int = 20,  # Use gpt oss 20b as baseline
    base_active_params: float = 3.6,
    base_cost_per_million_output_tokens: float = 0.50,
    base_cost_per_million_input_tokens: float = 0.07,
):
    """
    Estimates the relative cost of an LLM interaction.
    Score combines:
    - Quadratic prefill cost from processing the prompt tokens (assuming no caching) in attention layers
        + Linear cost from processing prompt tokens in MoE/Dense MLP layers.
    - Decode cost scaled by halfway point of decoded tokens + prompt length.
    - Slight exponential at Model size to penalize the size of server and hardware needed.

    Sums across all completions in a single problem request to handle the overhead of tool calling.

    Args:
        - model_config: The ModelConfig interface containing how many total and active params a model has.
        - inference_config: The Inference config interface containing the list of completions required to resolve problem.
        - attention_fraction: The fraction of weights that are attention weights. These tend to be roughly 1/6 of total active.
    """

    def io_cost(model_config: ModelConfig, completion_config: CompletionConfig):
        # Scale of the model based on active parameters
        param_scale = model_config.active_params / base_active_params
        # Parameter scaling: attention scales quadratically, remainder linearly
        param_factor = (attention_fraction * (param_scale**2)) + ((1 - attention_fraction) * param_scale)

        # Hardware penalty: slight exponential based on total params to penalize server size
        total_param_scale = model_config.total_params / base_total_params
        hardware_penalty = total_param_scale**1.2

        overall_model_factor = param_factor * hardware_penalty

        # Image tokens occupy the prompt alongside text tokens (they drive the
        # same quadratic attention prefill and lengthen the decode sequence).
        image_tokens = getattr(completion_config, "image_tokens", 0) or 0
        total_prompt_tokens = completion_config.prompt_tokens + image_tokens

        # Token counts in millions to match base costs
        p_m = total_prompt_tokens / 1_000_000
        c_m = completion_config.completion_tokens / 1_000_000

        # Prefill cost: Quadratic for attention layers, linear for MoE/Dense MLP
        prefill_token_factor = (attention_fraction * (p_m**2)) + ((1 - attention_fraction) * p_m)
        input_cost = base_cost_per_million_input_tokens * prefill_token_factor * overall_model_factor

        # Image-token surcharge: charge image tokens at IMAGE_TOKEN_PREMIUM x the
        # linear input rate (their quadratic share is already in input_cost above).
        image_surcharge = (
            base_cost_per_million_input_tokens
            * (image_tokens / 1_000_000)
            * (IMAGE_TOKEN_PREMIUM - 1.0)
            * overall_model_factor
        )

        # Decode cost: scaled by average sequence length during decode (in millions)
        avg_seq_len_m = (total_prompt_tokens + (completion_config.completion_tokens / 2)) / 1_000_000
        output_cost = base_cost_per_million_output_tokens * c_m * avg_seq_len_m * overall_model_factor

        return input_cost + image_surcharge + output_cost

    return sum(io_cost(model_config, comp) for comp in inference_config.completions)


def load_model_config(model_id: str) -> ModelConfig:
    """
    Load one model config from configs/models.yaml.
    """

    config_path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    with open(config_path, "r") as f:
        models_data = yaml.safe_load(f)

    if model_id not in models_data:
        raise KeyError(f"Unknown model_id: {model_id}")

    model_info = models_data[model_id]
    return ModelConfig(
        id=model_id,
        source=model_info.get("source", ""),
        source_url=model_info.get("source_url", ""),
        total_params=model_info.get("total_params", 0),
        active_params=model_info.get("active_params", 0),
    )


def calculate_model_call_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    image_tokens: int = 0,
) -> float:
    """
    Convenience helper for costing a single model completion.
    """

    model_config = load_model_config(model_id)
    inference_config = InferenceConfig(
        completions=[
            CompletionConfig(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                image_tokens=image_tokens,
            )
        ]
    )
    return calculate_cost(model_config, inference_config)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    with open(config_path, "r") as f:
        models_data = yaml.safe_load(f)

    # Mock inference config: 1 million input, 1 million output
    comp_config = CompletionConfig(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    inf_config = InferenceConfig(completions=[comp_config])
    print(f"{'Model ID':<35} | {'Cost ($)'}")
    print("-" * 50)
    for model_id, model_info in models_data.items():
        m_config = ModelConfig(
            id=model_id,
            source=model_info.get("source", ""),
            source_url=model_info.get("source_url", ""),
            total_params=model_info.get("total_params", 0),
            active_params=model_info.get("active_params", 0),
        )

        cost = calculate_cost(m_config, inf_config)
        print(f"{model_id:<35} | ${cost:.4f}")
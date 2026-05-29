from interfaces import InferenceConfig, ModelConfig

def calculate_cost(
    model_config: ModelConfig, 
    inference_config: InferenceConfig, 
    attention_fraction: float = 0.17,
    scale: int = 8192
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
        - scale: Scale factor to avoid explosions.
    """
    model_size_factor = ((model_config.total_params + model_config.active_params) ** (3/2)) / scale
    total = 0.0
    for c in inference_config.completions:
        mlp_fraction = 1.0 - attention_fraction
        prefill_linear = c.prompt_tokens * mlp_fraction
        prefill_attention = (c.prompt_tokens ** 2) * attention_fraction
        decode_context_penalty = c.prompt_tokens + (c.completion_tokens // 2)
        total += prefill_attention + prefill_linear + decode_context_penalty
    return total * model_size_factor

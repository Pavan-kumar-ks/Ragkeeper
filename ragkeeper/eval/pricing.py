# Best-effort USD price per 1M tokens. Verify against https://console.groq.com/settings/billing
# before trusting cost figures for anything beyond rough, relative comparisons.
PRICE_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    prices = PRICE_PER_MILLION_TOKENS.get(model)
    if prices is None:
        return None
    return (prompt_tokens * prices["input"] + completion_tokens * prices["output"]) / 1_000_000

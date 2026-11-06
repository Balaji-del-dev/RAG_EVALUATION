from decimal import Decimal


def compute_cost(config, prompt_tokens: int, completion_tokens: int) -> tuple[Decimal, Decimal]:
    """Cost comes from the prices on the RagConfig. Local models are priced at 0 unless a config sets
    a price, for example to estimate what the same traffic would cost on a paid API."""
    thousand = Decimal(1000)
    return (
        Decimal(prompt_tokens) / thousand * config.input_cost_per_1k,
        Decimal(completion_tokens) / thousand * config.output_cost_per_1k,
    )

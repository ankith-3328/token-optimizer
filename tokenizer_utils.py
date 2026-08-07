import tiktoken
import config

try:
    ENCODING = tiktoken.encoding_for_model("gpt-4")
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def count_tokens_for(text: str, encoding: str) -> int:
    encoder = tiktoken.get_encoding(encoding)
    return len(encoder.encode(text))

def estimate_cost(
    n_tokens: int,
    model: str | None = None,
    output: bool = False,
) -> float:
    model = model or config.COST_MODEL

    if model not in config.PRICING:
        raise KeyError(f"No pricing configured for {model!r}")

    input_price, output_price = config.PRICING[model]
    price = output_price if output else input_price

    return (n_tokens / 1_000_000) * price

def cost_saved(
    before_tokens: int,
    after_tokens: int,
    model: str | None = None,
) -> float:
    return (
        estimate_cost(before_tokens, model)
        - estimate_cost(after_tokens, model)
    )

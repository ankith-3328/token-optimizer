import tiktoken

try:
    ENCODING = tiktoken.encoding_for_model("gpt-4")
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))
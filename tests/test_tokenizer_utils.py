from tokenizer_utils import count_tokens


def test_count_tokens_non_empty():
    assert count_tokens("hello world") > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0
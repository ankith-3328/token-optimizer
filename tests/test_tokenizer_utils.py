from tokenizer_utils import count_tokens


def test_count_tokens_non_empty():
    assert count_tokens("hello world") > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_known_text():
    assert count_tokens("hello world") == 2


def test_count_tokens_longer_text():
    short = "hello"
    long = "hello world this is a much longer sentence"

    assert count_tokens(long) > count_tokens(short)


def test_count_tokens_special_characters():
    assert count_tokens("Hello! @#$%^&*() 123") > 0


def test_count_tokens_unicode():
    assert count_tokens("Hello नमस्ते 世界") > 0
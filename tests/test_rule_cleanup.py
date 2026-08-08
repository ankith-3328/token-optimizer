from stages.rule_cleanup import RuleCleanupStage


def test_removes_filler_phrases():
    stage = RuleCleanupStage()

    out = stage.run(
        "Please carefully review the code in order to find bugs."
    )

    assert "please carefully" not in out.lower()
    assert "in order to" not in out.lower()
    assert "review" in out.lower()
    assert "bugs" in out.lower()


def test_removes_single_filler_words():
    stage = RuleCleanupStage()

    out = stage.run(
        "Please just simply review this carefully."
    )

    lower = out.lower()

    assert "please" not in lower
    assert "just" not in lower
    assert "simply" not in lower
    assert "review" in lower


def test_collapses_multiple_spaces():
    stage = RuleCleanupStage()

    out = stage.run("Hello     world       again")

    assert out == "Hello world again"


def test_collapses_multiple_newlines():
    stage = RuleCleanupStage()

    out = stage.run("Line1\n\n\n\nLine2")

    assert out == "Line1\n\nLine2"


def test_removes_space_before_punctuation():
    stage = RuleCleanupStage()

    out = stage.run("Hello , world !")

    assert out == "Hello, world!"


def test_preserves_inline_code():
    stage = RuleCleanupStage()

    text = "Please review `print('hello')` carefully."

    out = stage.run(text)

    assert "`print('hello')`" in out


def test_preserves_fenced_code_block():
    stage = RuleCleanupStage()

    text = (
        "Please review\n\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
        "carefully."
    )

    out = stage.run(text)

    assert "```python" in out
    assert "print('hello')" in out
    assert "```" in out


def test_preserves_placeholders():
    stage = RuleCleanupStage()

    text = (
        "Please help {user_name} with the {issue_type} request."
    )

    out = stage.run(text)

    assert "{user_name}" in out
    assert "{issue_type}" in out


def test_leading_capitalization_is_preserved():
    stage = RuleCleanupStage()

    out = stage.run("Please review the code.")

    assert out.startswith("Review")


def test_empty_string_passthrough():
    stage = RuleCleanupStage()

    assert stage.run("") == ""


def test_whitespace_only_passthrough():
    stage = RuleCleanupStage()

    assert stage.run("     ") == "     "


def test_already_clean_text_is_unchanged():
    stage = RuleCleanupStage()

    text = "Review the code for bugs."

    assert stage.run(text) == text


def test_filler_inside_protected_region_not_removed():
    stage = RuleCleanupStage()

    text = "Please review `{please}` carefully."

    out = stage.run(text)

    assert "`{please}`" in out


def test_multiple_cleanup_operations_together():
    stage = RuleCleanupStage()

    text = (
        "Please carefully    review the code   "
        "in order to   find bugs  ."
    )

    out = stage.run(text)

    assert out == "Review the code find bugs."
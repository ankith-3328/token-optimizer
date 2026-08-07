from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    category: str
    text: str


LONG_CONTEXT_TEXT = " ".join(
    [
        (
            "The quick brown fox jumps over the lazy dog near the riverbank. "
            "Please carefully analyze the sentence structure and preserve all "
            "important semantic information while reducing redundancy."
        )
        for _ in range(150)
    ]
)


BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [

    # ------------------------------------------------------------------
    # Verbose prompts (4)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="verbose_code_review",
        category="verbose",
        text=(
            "Please carefully review the following code and provide a detailed "
            "analysis of potential bugs, performance bottlenecks, security "
            "vulnerabilities, edge cases, and areas where meaningful "
            "improvements can be made."
        ),
    ),

    BenchmarkPrompt(
        id="verbose_support_summary",
        category="verbose",
        text=(
            "Please thoroughly examine the customer support conversation below "
            "and generate a detailed summary that highlights key issues, "
            "customer concerns, actions taken, unresolved questions, and "
            "recommended next steps."
        ),
    ),

    BenchmarkPrompt(
        id="verbose_meeting_notes",
        category="verbose",
        text=(
            "Carefully review the meeting transcript and create a comprehensive "
            "summary containing major discussion points, action items, owners, "
            "deadlines, risks, dependencies, and follow-up tasks."
        ),
    ),

    BenchmarkPrompt(
        id="verbose_design_review",
        category="verbose",
        text=(
            "Analyze the proposed software architecture in detail and identify "
            "strengths, weaknesses, scalability concerns, maintainability "
            "issues, operational risks, and opportunities for improvement."
        ),
    ),

    # ------------------------------------------------------------------
    # Already optimized / terse prompts (3)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="terse_sentiment",
        category="already_terse",
        text="Classify the sentiment as positive, negative, or neutral.",
    ),

    BenchmarkPrompt(
        id="terse_translate",
        category="already_terse",
        text="Translate the text to French.",
    ),

    BenchmarkPrompt(
        id="terse_extract",
        category="already_terse",
        text="Extract all email addresses.",
    ),

    # ------------------------------------------------------------------
    # Few-shot prompts (3)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="fewshot_sentiment",
        category="fewshot",
        text=(
            "Determine sentiment.\n\n"
            "Review: I love this product.\n"
            "Sentiment: Positive\n\n"
            "Review: This was disappointing.\n"
            "Sentiment: Negative\n\n"
            "Review: It works as expected.\n"
            "Sentiment: Neutral\n\n"
            "Review: The experience exceeded expectations.\n"
            "Sentiment:"
        ),
    ),

    BenchmarkPrompt(
        id="fewshot_priority",
        category="fewshot",
        text=(
            "Assign priority.\n\n"
            "Ticket: Production outage.\n"
            "Priority: High\n\n"
            "Ticket: Login occasionally slow.\n"
            "Priority: Medium\n\n"
            "Ticket: Typo on settings page.\n"
            "Priority: Low\n\n"
            "Ticket: Database corruption detected.\n"
            "Priority:"
        ),
    ),

    BenchmarkPrompt(
        id="fewshot_category",
        category="fewshot",
        text=(
            "Categorize requests.\n\n"
            "Request: Reset my password.\n"
            "Category: Account\n\n"
            "Request: My payment failed.\n"
            "Category: Billing\n\n"
            "Request: API returns 500 errors.\n"
            "Category: Technical\n\n"
            "Request: Refund was not received.\n"
            "Category:"
        ),
    ),

    # ------------------------------------------------------------------
    # Placeholder prompts (3)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="placeholder_support",
        category="placeholder",
        text=(
            "You are helping {user_name} resolve a {issue_type} issue. "
            "Respond in a {tone} tone and limit the response to "
            "{max_words} words."
        ),
    ),

    BenchmarkPrompt(
        id="placeholder_email",
        category="placeholder",
        text=(
            "Write an email to {recipient_name} regarding {subject}. "
            "Use a {tone} tone and include reference number {ticket_id}."
        ),
    ),

    BenchmarkPrompt(
        id="placeholder_report",
        category="placeholder",
        text=(
            "Generate a report for {department}. "
            "Include metrics from {start_date} to {end_date} "
            "and summarize key findings."
        ),
    ),

    # ------------------------------------------------------------------
    # Negation / constraints (3)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="negation_pii",
        category="negation",
        text=(
            "Summarize the ticket below. Do not include PII. "
            "Never reveal customer email addresses or phone numbers."
        ),
    ),

    BenchmarkPrompt(
        id="constraint_json",
        category="negation",
        text=(
            "Return only valid JSON. Do not include explanations, "
            "markdown, comments, or additional text."
        ),
    ),

    BenchmarkPrompt(
        id="constraint_facts",
        category="negation",
        text=(
            "Answer using only the provided context. "
            "Do not invent facts and never speculate."
        ),
    ),

    # ------------------------------------------------------------------
    # Code blocks (2)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="code_review_python",
        category="code",
        text=(
            "Review the following code:\n\n"
            "```python\n"
            "def divide(a, b):\n"
            "    return a / b\n"
            "```\n\n"
            "Identify potential issues."
        ),
    ),

    BenchmarkPrompt(
        id="code_review_sql",
        category="code",
        text=(
            "Analyze the following SQL query:\n\n"
            "```sql\n"
            "SELECT * FROM users WHERE email = ?\n"
            "```\n\n"
            "Suggest improvements."
        ),
    ),

    # ------------------------------------------------------------------
    # Long-context stress test (1)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="long_context",
        category="long_context",
        text=LONG_CONTEXT_TEXT,
    ),
]
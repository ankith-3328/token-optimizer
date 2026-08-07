from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    category: str
    text: str


_LONG_PARAS = [
    (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "Please carefully analyze the sentence structure and preserve all "
        "important semantic information while reducing redundancy."
    ),
    (
        "In order to fully understand the passage above, kindly make sure that "
        "you take into account every relevant detail and do not overlook any "
        "constraint that might affect the final answer."
    ),
    (
        "Additionally, please be sure to note that the animal mentioned earlier "
        "is a fox, the obstacle is a dog, and the setting is a riverbank."
    ),
]

LONG_CONTEXT_TEXT = (
    " ".join(_LONG_PARAS[i % len(_LONG_PARAS)] for i in range(120))
    + "\n\nQuestion: Which animal jumps over the lazy dog? Answer briefly."
)


BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [

    # ------------------------------------------------------------------
    # Verbose prompts (4)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="verbose_code_review",
        category="verbose",
        text=(
            "Please carefully review the following code in order to provide a "
            "detailed analysis of any potential bugs, performance bottlenecks, "
            "security vulnerabilities, edge cases, and areas where meaningful "
            "improvements can be made. Kindly make sure that your response is "
            "thorough and well structured.\n\n"
            "```python\n"
            "def fetch_user(db, user_id):\n"
            "    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
            "    return db.execute(query).fetchone()\n"
            "```"
        ),
    ),

    BenchmarkPrompt(
        id="verbose_support_summary",
        category="verbose",
        text=(
            "Please thoroughly examine the customer support conversation below "
            "and generate a detailed summary that highlights key issues, "
            "customer concerns, actions taken, unresolved questions, and "
            "recommended next steps. Please make sure that you do not omit "
            "any important detail.\n\n"
            "Agent: Thanks for contacting support, how can I help you today?\n"
            "Customer: I was charged twice for order #48291 and need a refund.\n"
            "Agent: I've opened ticket T-9912 and escalated billing.\n"
            "Customer: Please resolve this before Friday."
        ),
    ),

    BenchmarkPrompt(
        id="verbose_meeting_notes",
        category="verbose",
        text=(
            "Carefully review the meeting transcript below and create a "
            "comprehensive summary containing major discussion points, action "
            "items, owners, deadlines, risks, dependencies, and follow-up "
            "tasks in order to keep everyone aligned.\n\n"
            "Alice: We need the API redesign ready by March 15.\n"
            "Bob: I'll own the migration plan and flag risks by Wednesday.\n"
            "Carol: Dependency on the auth service might slip the date."
        ),
    ),

    BenchmarkPrompt(
        id="verbose_design_review",
        category="verbose",
        text=(
            "Please analyze the proposed software architecture in detail and "
            "identify strengths, weaknesses, scalability concerns, "
            "maintainability issues, operational risks, and opportunities for "
            "improvement. Kindly ensure your feedback is actionable.\n\n"
            "Proposal: sync monolith writes to three regional Postgres replicas "
            "via a single Kafka topic named orders.events, with a 5-minute "
            "consumer lag SLO and no dead-letter queue yet."
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
            "Determine the sentiment of each customer review. Reply with "
            "exactly one of: Positive, Negative, or Neutral.\n\n"
            "Review: I absolutely love this product and would buy it again.\n"
            "Sentiment: Positive\n\n"
            "Review: This was deeply disappointing and a complete waste of money.\n"
            "Sentiment: Negative\n\n"
            "Review: It works as expected, nothing special to report either way.\n"
            "Sentiment: Neutral\n\n"
            "Review: Shipping was fine but the item itself feels low quality.\n"
            "Sentiment: Negative\n\n"
            "Review: The packaging was nice and customer support was helpful.\n"
            "Sentiment: Positive\n\n"
            "Review: The experience exceeded every expectation I had.\n"
            "Sentiment:"
        ),
    ),

    BenchmarkPrompt(
        id="fewshot_priority",
        category="fewshot",
        text=(
            "Assign a priority level to each support ticket. Use High, Medium, "
            "or Low based on business impact.\n\n"
            "Ticket: Production outage affecting all checkout traffic.\n"
            "Priority: High\n\n"
            "Ticket: Login page is occasionally slow for some users.\n"
            "Priority: Medium\n\n"
            "Ticket: Minor typo on the settings page footer text.\n"
            "Priority: Low\n\n"
            "Ticket: Nightly report email arrives one hour late.\n"
            "Priority: Low\n\n"
            "Ticket: Payment webhooks failing intermittently in EU region.\n"
            "Priority: High\n\n"
            "Ticket: Database corruption detected on the primary replica.\n"
            "Priority:"
        ),
    ),

    BenchmarkPrompt(
        id="fewshot_category",
        category="fewshot",
        text=(
            "Categorize each customer request. Labels: Account, Billing, "
            "Technical.\n\n"
            "Request: Please reset my password, I cannot sign in.\n"
            "Category: Account\n\n"
            "Request: My payment failed and I was charged twice somehow.\n"
            "Category: Billing\n\n"
            "Request: The API returns HTTP 500 errors on /v2/orders.\n"
            "Category: Technical\n\n"
            "Request: I need to update the email on my account profile.\n"
            "Category: Account\n\n"
            "Request: Invoice INV-2044 shows the wrong tax amount.\n"
            "Category: Billing\n\n"
            "Request: Refund was not received after the cancellation.\n"
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
    # Negation / constraints (4)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="negation_pii",
        category="negation",
        text=(
            "Summarize the ticket below. Do not include PII. "
            "Never reveal customer email addresses or phone numbers.\n\n"
            "Ticket: Jane Doe (jane.doe@example.com, +1-555-0100) reports "
            "that invoice INV-88 never arrived."
        ),
    ),

    BenchmarkPrompt(
        id="negation_contraction",
        category="negation",
        text=(
            "Draft a public status update about the outage. Don't mention "
            "internal hostnames and can't disclose customer names. "
            "Won't include speculative root-cause theories."
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
        id="constraint_budget",
        category="negation",
        text=(
            "Please make sure that the response you write is limited to "
            "exactly 250 words and costs no more than $1.50 per request. "
            "Keep confidence at or above 95%."
        ),
    ),

    # ------------------------------------------------------------------
    # Code blocks (2)
    # ------------------------------------------------------------------

    BenchmarkPrompt(
        id="code_review_python",
        category="code",
        text=(
            "Review the following code and identify potential issues:\n\n"
            "```python\n"
            "def divide(a, b):\n"
            "    # TODO: callers assume this never raises\n"
            "    return a / b\n"
            "\n"
            "def safe_mean(values):\n"
            "    total = 0\n"
            "    for value in values:\n"
            "        total += value\n"
            "    return total / len(values)  # ZeroDivisionError if empty\n"
            "\n"
            "def load_config(path):\n"
            "    with open(path) as handle:\n"
            "        return eval(handle.read())  # unsafe\n"
            "```\n\n"
            "Focus on correctness and security."
        ),
    ),

    BenchmarkPrompt(
        id="code_review_sql",
        category="code",
        text=(
            "Analyze the following SQL query and suggest improvements:\n\n"
            "```sql\n"
            "-- intentional full scan for debugging; do not ship\n"
            "SELECT u.id, u.email, o.total\n"
            "FROM users u\n"
            "LEFT JOIN orders o ON o.user_id = u.id\n"
            "WHERE u.email = ?\n"
            "  AND o.created_at > '2024-01-01'\n"
            "ORDER BY o.total DESC;\n"
            "```\n\n"
            "Call out indexing and injection risks."
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

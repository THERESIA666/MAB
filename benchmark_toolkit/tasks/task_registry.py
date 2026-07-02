"""Task registry — defines the standard benchmark task suite."""

from benchmark_toolkit.core.task import Task, TaskInput, TaskSuite


# ══════════════════════════════════════════════════════════════════════
# Task 1: Multi-hop Question Answering
# ══════════════════════════════════════════════════════════════════════

MULTI_HOP_QUESTIONS = [
    {
        "id": "mhq-1",
        "question": "What is the official language of the country where the headquarters "
                     "of the company that created AlphaGo is located?",
        "answer": "English",
        "chain": "AlphaGo → DeepMind (UK) → Google acquired it → USA or UK. "
                  "DeepMind HQ is in London, UK → official language is English.",
        "keywords": ["english", "england", "uk", "britain", "british"],
    },
    {
        "id": "mhq-2",
        "question": "In what year was the first person to walk on the Moon born?",
        "answer": "1930",
        "chain": "First person on Moon → Neil Armstrong → born 1930.",
        "keywords": ["1930"],
    },
    {
        "id": "mhq-3",
        "question": "What is the chemical symbol of the element discovered by the scientist "
                     "who also developed the theory of relativity?",
        "answer": "Es",
        "chain": "Theory of relativity → Albert Einstein → Einsteinium → Es.",
        "keywords": ["es", "einsteinium"],
    },
]

def _make_multihop_task(entry: dict) -> Task:
    def input_generator() -> TaskInput:
        return TaskInput(
            task_id=entry["id"],
            description=f"Answer this multi-step question: {entry['question']}",
            input_data={"question": entry["question"]},
            context="You must reason step by step. Each step depends on the previous one. "
                     "Do NOT guess — think through each step carefully.",
        )

    def evaluator(output: str) -> float:
        output_lower = output.lower()
        keywords = entry["keywords"]
        matches = sum(1 for kw in keywords if kw in output_lower)
        base_score = matches / len(keywords) if keywords else 0.0

        # Penalize if output is too short (likely incomplete)
        if len(output) < 20:
            base_score *= 0.5

        # Bonus for showing reasoning steps
        reasoning_indicators = ["step", "first", "then", "therefore", "because", "→"]
        has_reasoning = any(ind in output_lower for ind in reasoning_indicators)
        if has_reasoning:
            base_score = min(1.0, base_score + 0.2)

        return min(1.0, base_score)

    return Task(
        task_id=entry["id"],
        name=f"Multi-hop QA: {entry['question'][:60]}...",
        description=entry["question"],
        category="qa",
        difficulty="medium",
        input_generator=input_generator,
        evaluator=evaluator,
        min_agents_recommended=2,
        expected_duration_seconds=45,
        tags=["reasoning", "knowledge", "multi-step"],
        ground_truth=entry["answer"],
    )


# ══════════════════════════════════════════════════════════════════════
# Task 2: Code Generation + Review
# ══════════════════════════════════════════════════════════════════════

CODING_PROBLEMS = [
    {
        "id": "code-1",
        "problem": "Write a Python function `is_palindrome(s: str) -> bool` that returns "
                    "True if the input string is a palindrome (reads the same forwards and "
                    "backwards), ignoring case, spaces, and punctuation. "
                    "Include docstring and handle edge cases.",
        "test_cases": [
            ("is_palindrome('A man a plan a canal Panama')", True),
            ("is_palindrome('race a car')", False),
            ("is_palindrome('')", True),
            ("is_palindrome('a')", True),
            ("is_palindrome('No lemon, no melon')", True),
            ("is_palindrome('hello')", False),
        ],
        "key_requirements": [
            "function named is_palindrome",
            "handles spaces",
            "handles punctuation",
            "handles case-insensitivity",
            "handles empty string",
            "has docstring",
        ],
    },
    {
        "id": "code-2",
        "problem": "Write a Python function `fibonacci(n: int) -> list[int]` that returns "
                    "the first n Fibonacci numbers. Include input validation (n must be "
                    "non-negative integer) and a docstring. Handle n=0 and n=1 correctly.",
        "test_cases": [
            ("fibonacci(0)", []),
            ("fibonacci(1)", [0]),
            ("fibonacci(2)", [0, 1]),
            ("fibonacci(5)", [0, 1, 1, 2, 3]),
            ("fibonacci(10)[-1]", 34),
        ],
        "key_requirements": [
            "function named fibonacci",
            "input validation",
            "handles n=0",
            "handles n=1",
            "correct sequence",
            "has docstring",
        ],
    },
]

def _make_coding_task(entry: dict) -> Task:
    def input_generator() -> TaskInput:
        return TaskInput(
            task_id=entry["id"],
            description=f"Write code for: {entry['problem']}",
            input_data={"problem": entry["problem"]},
            context="One agent should write the initial code. Another should review it for "
                     "correctness, edge cases, and code quality. A third should produce the "
                     "final reviewed version. Only output the final Python code.",
        )

    def evaluator(output: str) -> float:
        score = 0.0
        max_score = len(entry["key_requirements"]) + len(entry["test_cases"])

        # Check key requirements
        output_lower = output.lower()
        for req in entry["key_requirements"]:
            keyword = req.split()[0] if req.split() else req  # crude keyword extraction
            if keyword.lower() in output_lower or req.lower() in output_lower:
                score += 1.0

        # Extract and test code
        code = _extract_code(output)
        if code:
            for test_expr, expected in entry["test_cases"]:
                try:
                    exec(code)
                    result = eval(test_expr)
                    if result == expected:
                        score += 1.0
                except Exception:
                    pass

        return min(1.0, score / max_score)

    return Task(
        task_id=entry["id"],
        name=f"Code: {entry['problem'][:55]}...",
        description=entry["problem"],
        category="code_generation",
        difficulty="medium",
        input_generator=input_generator,
        evaluator=evaluator,
        min_agents_recommended=2,
        expected_duration_seconds=90,
        tags=["code", "review", "testing"],
    )


def _extract_code(output: str) -> str:
    """Extract Python code from markdown code blocks or plain text."""
    import re
    # Try markdown code block first
    match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find function definition
    match = re.search(r'(def\s+\w+.*?)(?:\n\n|\Z)', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output.strip()


# ══════════════════════════════════════════════════════════════════════
# Task 3: Data Analysis
# ══════════════════════════════════════════════════════════════════════

DATA_ANALYSIS_TASKS = [
    {
        "id": "data-1",
        "scenario": "You are given sales data for a small online store for the past week:\n\n"
                     "| Day | Product | Units Sold | Price per Unit | Customer Region |\n"
                     "|-----|---------|------------|----------------|------------------|\n"
                     "| Mon | Widget A | 12 | $15.00 | US |\n"
                     "| Mon | Widget B | 5 | $25.00 | EU |\n"
                     "| Tue | Widget A | 8 | $15.00 | US |\n"
                     "| Tue | Widget B | 11 | $25.00 | EU |\n"
                     "| Wed | Widget A | 15 | $15.00 | US |\n"
                     "| Wed | Widget B | 3 | $25.00 | Asia |\n"
                     "| Thu | Widget A | 10 | $15.00 | US |\n"
                     "| Thu | Widget B | 9 | $25.00 | EU |\n"
                     "| Fri | Widget A | 20 | $15.00 | US |\n"
                     "| Fri | Widget B | 14 | $25.00 | Asia |\n"
                     "| Sat | Widget A | 18 | $15.00 | US |\n"
                     "| Sat | Widget B | 7 | $25.00 | EU |\n"
                     "| Sun | Widget A | 5 | $15.00 | Asia |\n"
                     "| Sun | Widget B | 2 | $25.00 | Asia |",
        "question": "Analyze this sales data and report:\n"
                     "1. Total revenue for the week\n"
                     "2. Best-selling product by units\n"
                     "3. Best-selling product by revenue\n"
                     "4. Region with highest total sales\n"
                     "5. Day with highest revenue\n"
                     "6. Any trends or insights",
        "expected_values": {
            "total_revenue": 2680.00,
            "best_product_units": "Widget A",
            "best_product_revenue": "Widget A",
            "best_region": "US",
            "best_day": "Friday",
        },
    },
]

def _make_data_task(entry: dict) -> Task:
    def input_generator() -> TaskInput:
        return TaskInput(
            task_id=entry["id"],
            description=entry["question"],
            input_data={"scenario": entry["scenario"], "question": entry["question"]},
            context="This requires multi-step data analysis. One agent should understand "
                     "the data structure. Another should perform calculations. A third "
                     "should write the final report.",
        )

    def evaluator(output: str) -> float:
        output_lower = output.lower()
        expected = entry["expected_values"]
        score = 0.0
        checks = 5

        # Check each expected value
        if str(expected["total_revenue"]) in output or "2680" in output:
            score += 1.0
        if expected["best_product_units"].lower() in output_lower:
            score += 1.0
        if expected["best_product_revenue"].lower() in output_lower:
            score += 1.0
        if expected["best_region"].lower() in output_lower:
            score += 1.0
        if expected["best_day"].lower() in output_lower or "fri" in output_lower:
            score += 1.0

        # Bonus for showing calculations
        if any(word in output_lower for word in ["calculate", "sum", "total", "average"]):
            score += 0.5

        return min(1.0, score / checks)

    return Task(
        task_id=entry["id"],
        name="Data Analysis: Online Store Sales",
        description=entry["question"],
        category="data_analysis",
        difficulty="medium",
        input_generator=input_generator,
        evaluator=evaluator,
        min_agents_recommended=2,
        expected_duration_seconds=90,
        tags=["data", "analysis", "calculation", "reporting"],
    )


# ══════════════════════════════════════════════════════════════════════
# Task 4: Coordinated Tool Use / Research
# ══════════════════════════════════════════════════════════════════════

RESEARCH_TASKS = [
    {
        "id": "research-1",
        "topic": "Compare the transformer architecture (introduced in 'Attention is All "
                  "You Need') with the Mamba architecture (State Space Models). "
                  "Explain the key differences in terms of:\n"
                  "1. Computational complexity\n"
                  "2. Memory requirements\n"
                  "3. Strengths and weaknesses of each\n"
                  "4. Best use cases for each",
        "key_points": [
            "quadratic", "linear", "attention", "state space",
            "long sequence", "parallel", "recurrent",
        ],
    },
    {
        "id": "research-2",
        "topic": "Explain the concept of 'speculative decoding' in large language models. "
                  "Cover:\n"
                  "1. What problem it solves\n"
                  "2. How it works at a high level\n"
                  "3. Typical speedup factors\n"
                  "4. Trade-offs and limitations",
        "key_points": [
            "draft model", "target model", "verification", "speedup",
            "latency", "throughput", "speculative",
        ],
    },
]

def _make_research_task(entry: dict) -> Task:
    def input_generator() -> TaskInput:
        return TaskInput(
            task_id=entry["id"],
            description=entry["topic"],
            input_data={"topic": entry["topic"]},
            context="Multiple agents should collaborate: one researches the topic, "
                     "another organizes the information, and a third writes a "
                     "comprehensive report. Be thorough but concise.",
        )

    def evaluator(output: str) -> float:
        output_lower = output.lower()
        key_points = entry["key_points"]
        matches = sum(1 for kp in key_points if kp in output_lower)
        score = matches / len(key_points)

        # Reward structure and organization
        if len(output) > 200:
            score = min(1.0, score + 0.1)

        # Reward numbered/bulleted lists (indicates organization)
        import re
        if re.search(r'(\d\.|\*|\-)\s', output):
            score = min(1.0, score + 0.1)

        return min(1.0, score)

    return Task(
        task_id=entry["id"],
        name=f"Research: {entry['topic'][:55]}...",
        description=entry["topic"],
        category="tool_use",
        difficulty="hard",
        input_generator=input_generator,
        evaluator=evaluator,
        min_agents_recommended=2,
        expected_duration_seconds=90,
        tags=["research", "synthesis", "knowledge"],
    )


# ══════════════════════════════════════════════════════════════════════
# Task 5: Planning + Execution
# ══════════════════════════════════════════════════════════════════════

PLANNING_TASKS = [
    {
        "id": "plan-1",
        "scenario": "You are organizing a 3-day tech conference for 500 attendees. "
                     "Plan the following:\n"
                     "1. Venue requirements\n"
                     "2. Schedule structure (keynotes, workshops, breaks)\n"
                     "3. Catering plan\n"
                     "4. Technical requirements (AV, WiFi, recording)\n"
                     "5. Budget estimate breakdown",
        "key_elements": [
            "venue", "schedule", "catering", "technical", "budget",
            "wifi", "keynote", "workshop", "break",
        ],
    },
]

def _make_planning_task(entry: dict) -> Task:
    def input_generator() -> TaskInput:
        return TaskInput(
            task_id=entry["id"],
            description=entry["scenario"],
            input_data={"scenario": entry["scenario"]},
            context="This requires systematic planning. One agent should outline the "
                     "structure, another should fill in details for each section, and "
                     "a third should review for completeness and consistency.",
        )

    def evaluator(output: str) -> float:
        output_lower = output.lower()
        elements = entry["key_elements"]
        matches = sum(1 for e in elements if e in output_lower)
        score = matches / len(elements)

        # Bonus for quantitative reasoning (numbers for budget, attendees, etc.)
        import re
        numbers = re.findall(r'\b\d{2,}\b', output)
        if len(numbers) >= 3:
            score = min(1.0, score + 0.1)

        return min(1.0, score)

    return Task(
        task_id=entry["id"],
        name="Planning: Tech Conference Organization",
        description=entry["scenario"],
        category="tool_use",
        difficulty="medium",
        input_generator=input_generator,
        evaluator=evaluator,
        min_agents_recommended=2,
        expected_duration_seconds=90,
        tags=["planning", "organization", "coordination"],
    )


# ══════════════════════════════════════════════════════════════════════
# Suite factory
# ══════════════════════════════════════════════════════════════════════

def create_default_suite() -> TaskSuite:
    """Create the default benchmark task suite."""
    tasks = []

    # Multi-hop QA tasks
    for entry in MULTI_HOP_QUESTIONS:
        tasks.append(_make_multihop_task(entry))

    # Coding tasks
    for entry in CODING_PROBLEMS:
        tasks.append(_make_coding_task(entry))

    # Data analysis tasks
    for entry in DATA_ANALYSIS_TASKS:
        tasks.append(_make_data_task(entry))

    # Research tasks
    for entry in RESEARCH_TASKS:
        tasks.append(_make_research_task(entry))

    # Planning tasks
    for entry in PLANNING_TASKS:
        tasks.append(_make_planning_task(entry))

    return TaskSuite(tasks)

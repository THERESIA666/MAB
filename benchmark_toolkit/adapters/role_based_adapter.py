"""Role-Based Collaboration adapter — implements the CrewAI orchestration pattern.

CrewAI uses a role-based model where agents have distinct roles, goals, and
backstories, and tasks are executed sequentially with explicit delegation and
expected_output specifications.

This adapter implements the same pattern through our InstrumentedLLM,
capturing the essential CrewAI-style collaboration while being fully
compatible with our metrics collection (which the real CrewAI library
would bypass).

Key differences from MultiStepAdapter:
  - Agents have full CrewAI-style backstories and delegation capabilities
  - Task descriptions include expected_output fields
  - The Researcher can delegate sub-tasks to other agents
  - Follows CrewAI's sequential execution model with handoff validation
"""

from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput


class RoleBasedAdapter(BaseAdapter):
    """Implements CrewAI-style role-based sequential collaboration.

    This adapter uses the same three roles (Researcher, Analyst, Writer) but
    with CrewAI-specific features: rich backstories, delegation, expected_output,
    and sequential task handoff with validation.
    """

    def __init__(self, config: AdapterConfig = None):
        if config is None:
            config = AdapterConfig(
                adapter_name="role-based",
                llm_model="deepseek-v4-flash",
                llm_provider="deepseek",
            )
        super().__init__(config)

    def solve(self, task_input: TaskInput) -> str:
        prompt = self._build_prompt(task_input)

        # Stage 1: Researcher with CrewAI-style role definition
        research = self.llm.chat(
            messages=[{
                "role": "system",
                "content": (
                    "You are a Research Specialist. Your role is to gather, "
                    "organize, and verify all relevant information for assigned tasks.\n\n"
                    "Backstory: You have years of experience in investigative research "
                    "and can quickly identify the most relevant facts, data points, "
                    "and details needed to solve complex problems. You are thorough "
                    "and never skip steps in your reasoning.\n\n"
                    "Goal: Produce comprehensive research findings with all relevant "
                    "information, clearly organized and factually verified.\n\n"
                    "You may delegate sub-questions to other researchers if needed, "
                    "but you must validate all findings before passing them on."
                ),
            }, {
                "role": "user",
                "content": (
                    f"RESEARCH TASK:\n{prompt}\n\n"
                    "Expected Output: Detailed research findings covering all aspects "
                    "of the task. Include specific facts, logical chains, and relevant "
                    "context. Be comprehensive — the Analyst will depend on your work."
                ),
            }],
            agent_name="Researcher",
            purpose="research",
        )

        # Stage 2: Analyst with delegation capability
        analysis = self.llm.chat(
            messages=[{
                "role": "system",
                "content": (
                    "You are a Senior Analyst. Your role is to synthesize research "
                    "findings, identify key patterns and insights, and prepare "
                    "structured analysis for the Writer.\n\n"
                    "Backstory: With a background in data science and critical thinking, "
                    "you excel at extracting meaningful patterns from complex information. "
                    "You are known for catching inconsistencies and identifying the most "
                    "important insights.\n\n"
                    "Goal: Produce a structured analysis that highlights key insights, "
                    "verifies factual claims, and organizes information for clear communication.\n\n"
                    "If the research is incomplete on any point, note it explicitly rather "
                    "than guessing."
                ),
            }, {
                "role": "user",
                "content": (
                    f"ANALYSIS TASK:\n\n"
                    f"=== RESEARCH FINDINGS ===\n{research}\n\n"
                    f"=== ORIGINAL TASK ===\n{prompt}\n\n"
                    "Expected Output: Structured analysis with:\n"
                    "1. Key insights and patterns identified\n"
                    "2. Factual verification of research claims\n"
                    "3. Any gaps or inconsistencies noted\n"
                    "4. Organized summary ready for the Writer"
                ),
            }],
            agent_name="Analyst",
            purpose="analysis",
        )

        # Stage 3: Writer produces final output
        final = self.llm.chat(
            messages=[{
                "role": "system",
                "content": (
                    "You are a Technical Writer. Your role is to produce the final, "
                    "polished output based on the analysis provided.\n\n"
                    "Backstory: You specialize in translating complex technical analysis "
                    "into clear, well-structured, and accurate final deliverables. "
                    "Your work is known for being complete, well-organized, and free of errors.\n\n"
                    "Goal: Produce a final answer that is complete, accurate, clearly "
                    "structured, and directly addresses the original task. This is the "
                    "deliverable that will be evaluated."
                ),
            }, {
                "role": "user",
                "content": (
                    f"WRITING TASK:\n\n"
                    f"=== ANALYSIS ===\n{analysis}\n\n"
                    f"=== ORIGINAL TASK ===\n{prompt}\n\n"
                    "Expected Output: The final, polished answer. Must be:\n"
                    "- Complete: addresses all parts of the original task\n"
                    "- Accurate: all claims are verified against the analysis\n"
                    "- Well-structured: organized logically with clear sections\n"
                    "- Self-contained: can be understood without reading the research"
                ),
            }],
            agent_name="Writer",
            purpose="final_output",
        )

        return final

    def agent_names(self) -> list[str]:
        return ["Researcher", "Analyst", "Writer"]

    def _build_prompt(self, task_input: TaskInput) -> str:
        parts = [f"Task: {task_input.description}"]
        if task_input.context:
            parts.append(f"\nContext:\n{task_input.context}")
        if task_input.input_data:
            import json
            parts.append(
                f"\nInput Data:\n```json\n"
                f"{json.dumps(task_input.input_data, indent=2)}\n```"
            )
        return "\n".join(parts)

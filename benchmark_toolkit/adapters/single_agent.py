"""Single-agent baseline adapter — no multi-agent collaboration.

This serves as the control group: if a single LLM can solve the task just as
well as a multi-agent system, the collaboration overhead isn't justified.
"""

from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput


class SingleAgentAdapter(BaseAdapter):
    """A baseline adapter that uses a single LLM call (no multi-agent).

    This is the simplest possible approach — give the entire task to one
    LLM call and see what happens. It provides a lower bound for comparing
    multi-agent frameworks against.
    """

    def __init__(self, config: AdapterConfig = None):
        if config is None:
            config = AdapterConfig(
                adapter_name="single-agent",
                llm_model="claude-sonnet-4-6-20251001",
            )
        super().__init__(config)

    def solve(self, task_input: TaskInput) -> str:
        prompt = self._build_prompt(task_input)
        return self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            agent_name="single_agent",
            purpose="task_solving",
        )

    def agent_names(self) -> list[str]:
        return ["single_agent"]

    def _build_prompt(self, task_input: TaskInput) -> str:
        """Build a comprehensive prompt for the single agent."""
        parts = [
            f"# Task: {task_input.description}",
        ]
        if task_input.context:
            parts.append(f"\n## Context\n{task_input.context}")
        if task_input.input_data:
            parts.append(f"\n## Input Data\n```json\n{task_input.input_data}\n```")
        parts.append("\n## Instructions\n")
        parts.append("Please complete this task thoroughly. Provide your final answer clearly.")
        return "\n".join(parts)

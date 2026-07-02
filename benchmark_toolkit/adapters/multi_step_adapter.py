"""Multi-step collaboration adapter — simulates multi-agent workflow.

This adapter implements a research -> analyze -> write pipeline using
sequential LLM calls through the instrumented LLM. It captures the
collaboration pattern of a multi-agent system without requiring a
specific multi-agent framework to be installed.

This serves as a "middle ground" between single-agent baseline and
full multi-agent frameworks for benchmarking purposes.
"""

from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput


class MultiStepAdapter(BaseAdapter):
    """A multi-step collaboration adapter (3-stage: research -> analyze -> write).

    Unlike the single-agent adapter which makes one LLM call, this adapter
    implements a 3-stage collaborative pipeline:
      1. Research Agent: Gathers and organizes relevant information
      2. Analyst Agent: Synthesizes findings and extracts insights
      3. Writer Agent: Produces the final polished answer

    This captures the essential collaboration pattern of multi-agent systems
    (role specialization, sequential handoffs, iterative refinement) while
    being framework-agnostic.
    """

    def __init__(self, config: AdapterConfig = None):
        if config is None:
            config = AdapterConfig(
                adapter_name="multi-step",
                llm_model="deepseek-chat",
                llm_provider="deepseek",
            )
        super().__init__(config)

    def solve(self, task_input: TaskInput) -> str:
        prompt = self._build_prompt(task_input)

        # Stage 1: Research
        research = self.llm.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"You are a Research Specialist. Your job is to gather and "
                    f"organize all relevant information for this task.\n\n"
                    f"TASK:\n{prompt}\n\n"
                    f"Provide comprehensive research findings. Include specific "
                    f"facts, data points, and relevant details. Be thorough."
                ),
            }],
            agent_name="Researcher",
            purpose="research",
        )

        # Stage 2: Analysis
        analysis = self.llm.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"You are a Senior Analyst. Review the research findings below "
                    f"and provide a structured analysis.\n\n"
                    f"=== RESEARCH FINDINGS ===\n{research}\n\n"
                    f"=== ORIGINAL TASK ===\n{prompt}\n\n"
                    f"Identify key insights, verify facts, note any gaps, and "
                    f"prepare a clear analytical summary with supporting evidence."
                ),
            }],
            agent_name="Analyst",
            purpose="analysis",
        )

        # Stage 3: Final output
        final = self.llm.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"You are a Technical Writer. Based on the analysis below, "
                    f"produce the final polished answer.\n\n"
                    f"=== ANALYSIS ===\n{analysis}\n\n"
                    f"=== ORIGINAL TASK ===\n{prompt}\n\n"
                    f"Write a clear, complete, well-structured final answer. "
                    f"Make sure to include all key findings and present them "
                    f"in a organized manner. This is the FINAL output."
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
            parts.append(f"\nInput Data:\n```json\n{json.dumps(task_input.input_data, indent=2)}\n```")
        return "\n".join(parts)

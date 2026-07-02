"""CrewAI adapter — wraps CrewAI multi-agent framework.

CrewAI uses a role-based agent model with sequential or hierarchical task
execution. This adapter maps benchmark tasks to CrewAI agents and collects
metrics via the instrumented LLM.
"""

from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput


class CrewAIAdapter(BaseAdapter):
    """Adapter for the CrewAI multi-agent framework.

    CrewAI organizes agents into "crews" that execute tasks sequentially
    or in a hierarchy. Each agent has a role, goal, and backstory.

    Requires: pip install crewai
    """

    def __init__(self, config: AdapterConfig = None):
        if config is None:
            config = AdapterConfig(
                adapter_name="crewai",
                llm_model="claude-sonnet-4-6-20251001",
            )
        super().__init__(config)
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if CrewAI is installed."""
        try:
            import crewai  # noqa: F401
        except ImportError:
            raise ImportError(
                "CrewAI is not installed. Install it with: pip install crewai"
            )

    def solve(self, task_input: TaskInput) -> str:
        from crewai import Agent, Task, Crew

        # Parse task to determine appropriate agent configuration
        agents, crew_tasks = self._create_agents_and_tasks(task_input)

        # Create and run the crew
        crew = Crew(
            agents=agents,
            tasks=crew_tasks,
            verbose=False,
        )

        result = crew.kickoff()

        # CrewAI returns a TaskOutput-like object; extract string
        if hasattr(result, 'raw'):
            return str(result.raw)
        return str(result)

    def _create_agents_and_tasks(
        self, task_input: TaskInput
    ) -> tuple[list, list]:
        """Create CrewAI agents and tasks based on the benchmark task."""
        from crewai import Agent, Task

        prompt = self._build_crew_prompt(task_input)

        # Default multi-agent setup: Researcher + Analyst + Writer
        researcher = Agent(
            role="Research Specialist",
            goal=f"Gather and analyze all relevant information for: {task_input.description}",
            backstory="You are an expert researcher who excels at finding and "
                      "organizing information from multiple sources.",
            allow_delegation=True,
            verbose=False,
        )

        analyst = Agent(
            role="Senior Analyst",
            goal=f"Synthesize research findings and derive insights for: {task_input.description}",
            backstory="You are a senior analyst who can extract meaningful patterns "
                      "and insights from complex information.",
            allow_delegation=True,
            verbose=False,
        )

        writer = Agent(
            role="Technical Writer",
            goal="Produce a clear, accurate, and well-structured final answer",
            backstory="You are a meticulous technical writer who ensures outputs "
                      "are accurate, well-organized, and complete.",
            allow_delegation=False,
            verbose=False,
        )

        research_task = Task(
            description=f"Research task: {task_input.description}\n\n{prompt}",
            expected_output="Detailed research findings with all relevant information",
            agent=researcher,
        )

        analysis_task = Task(
            description="Analyze the research findings. Identify key insights, "
                        "patterns, and the most important information needed to "
                        "answer the original question.",
            expected_output="Structured analysis with key insights and supporting evidence",
            agent=analyst,
        )

        writing_task = Task(
            description="Write the final answer based on the analysis. Be clear, "
                        "concise, and thorough. Include all relevant findings.",
            expected_output="A comprehensive final answer to the original task",
            agent=writer,
        )

        return (
            [researcher, analyst, writer],
            [research_task, analysis_task, writing_task],
        )

    def agent_names(self) -> list[str]:
        return ["Researcher", "Analyst", "Writer"]

    def _build_crew_prompt(self, task_input: TaskInput) -> str:
        parts = [f"Task: {task_input.description}"]
        if task_input.context:
            parts.append(f"\nContext:\n{task_input.context}")
        if task_input.input_data:
            import json
            parts.append(f"\nInput Data:\n```json\n{json.dumps(task_input.input_data, indent=2)}\n```")
        return "\n".join(parts)

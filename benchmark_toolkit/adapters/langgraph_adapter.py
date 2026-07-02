"""LangGraph adapter — wraps LangChain's LangGraph for graph-based agent flows.

LangGraph represents agent workflows as directed graphs where nodes are
processing steps and edges define flow control. This adapter builds a
standard collaboration graph for benchmark tasks.
"""

from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangChain's LangGraph framework.

    LangGraph models agent workflows as state graphs with explicit
    routing logic. This makes it more structured than conversation-based
    approaches but requires more upfront design.

    Requires: pip install langgraph langchain langchain-anthropic
    """

    def __init__(self, config: AdapterConfig = None):
        if config is None:
            config = AdapterConfig(
                adapter_name="langgraph",
                llm_model="claude-sonnet-4-6-20251001",
            )
        super().__init__(config)
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import langgraph  # noqa: F401
        except ImportError:
            raise ImportError(
                "LangGraph is not installed. Install it with: "
                "pip install langgraph langchain langchain-anthropic"
            )

    def solve(self, task_input: TaskInput) -> str:
        import json
        from typing import TypedDict, Annotated
        from langgraph.graph import StateGraph, END
        from langgraph.graph.message import add_messages

        prompt = self._build_prompt(task_input)

        # Define state
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            research: str
            analysis: str
            final_answer: str
            stage: str

        # Node functions
        def research_node(state: AgentState) -> AgentState:
            research_prompt = (
                f"Research the following task thoroughly:\n\n{prompt}\n\n"
                "Identify all relevant facts, data points, and information "
                "needed to answer this question. Be comprehensive."
            )
            response = self.llm.chat(
                messages=[{"role": "user", "content": research_prompt}],
                agent_name="Researcher",
                purpose="research",
            )
            return {
                "research": response,
                "stage": "analysis",
                "messages": [{"role": "assistant", "content": f"[Research]: {response}"}],
            }

        def analysis_node(state: AgentState) -> AgentState:
            analysis_prompt = (
                f"Based on the following research findings, provide a detailed analysis:\n\n"
                f"=== RESEARCH ===\n{state.get('research', 'No research available')}\n\n"
                f"=== ORIGINAL TASK ===\n{prompt}\n\n"
                "Identify key insights, patterns, and determine the most important "
                "information needed to answer the original task."
            )
            response = self.llm.chat(
                messages=[{"role": "user", "content": analysis_prompt}],
                agent_name="Analyst",
                purpose="analysis",
            )
            return {
                "analysis": response,
                "stage": "writing",
                "messages": [{"role": "assistant", "content": f"[Analysis]: {response}"}],
            }

        def writing_node(state: AgentState) -> AgentState:
            writing_prompt = (
                f"Write the final answer based on the analysis below:\n\n"
                f"=== ANALYSIS ===\n{state.get('analysis', 'No analysis available')}\n\n"
                f"=== ORIGINAL TASK ===\n{prompt}\n\n"
                "Write a clear, complete, and well-structured final answer."
            )
            response = self.llm.chat(
                messages=[{"role": "user", "content": writing_prompt}],
                agent_name="Writer",
                purpose="final_output",
            )
            return {
                "final_answer": response,
                "stage": "done",
                "messages": [{"role": "assistant", "content": f"[Final Answer]: {response}"}],
            }

        def router(state: AgentState) -> str:
            stage = state.get("stage", "research")
            if stage == "done":
                return END
            return stage

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("research", research_node)
        workflow.add_node("analysis", analysis_node)
        workflow.add_node("writing", writing_node)

        workflow.set_entry_point("research")
        workflow.add_conditional_edges("research", router, {
            "analysis": "analysis",
            END: END,
        })
        workflow.add_conditional_edges("analysis", router, {
            "writing": "writing",
            END: END,
        })
        workflow.add_conditional_edges("writing", router, {
            END: END,
        })

        app = workflow.compile()

        # Execute
        initial_state = {
            "messages": [{"role": "user", "content": prompt}],
            "research": "",
            "analysis": "",
            "final_answer": "",
            "stage": "research",
        }

        final_state = app.invoke(initial_state)
        return final_state.get("final_answer", "No output produced.")

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

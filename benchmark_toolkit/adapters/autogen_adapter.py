"""AutoGen adapter — wraps Microsoft AutoGen multi-agent framework.

AutoGen uses a conversation-based model where agents chat with each other
to solve tasks. This adapter creates conversational agents and manages
the group chat flow.
"""

from benchmark_toolkit.core.adapter import BaseAdapter, AdapterConfig
from benchmark_toolkit.core.task import TaskInput


class AutoGenAdapter(BaseAdapter):
    """Adapter for Microsoft AutoGen multi-agent framework.

    AutoGen agents communicate through structured conversations. Tasks
    are solved through multi-turn dialogues between specialized agents.

    Requires: pip install pyautogen
    """

    def __init__(self, config: AdapterConfig = None):
        if config is None:
            config = AdapterConfig(
                adapter_name="autogen",
                llm_model="claude-sonnet-4-6-20251001",
            )
        super().__init__(config)
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import autogen  # noqa: F401
        except ImportError:
            raise ImportError(
                "AutoGen is not installed. Install it with: pip install pyautogen"
            )

    def solve(self, task_input: TaskInput) -> str:
        import autogen

        prompt = self._build_prompt(task_input)

        # Configure LLM for AutoGen
        llm_config = {
            "config_list": [{"model": self.config.llm_model, "api_key": self.config.resolve_api_key()}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.llm_provider == "anthropic":
            llm_config["config_list"][0]["api_type"] = "anthropic"

        # Create agents
        researcher = autogen.AssistantAgent(
            name="Researcher",
            system_message="You are a research specialist. Your job is to gather "
                           "and analyze all relevant information. Ask clarifying "
                           "questions when needed. After completing your research, "
                           "summarize your findings clearly.",
            llm_config=llm_config,
        )

        analyst = autogen.AssistantAgent(
            name="Analyst",
            system_message="You are a senior analyst. Take the researcher's findings, "
                           "identify key patterns and insights, and prepare a structured "
                           "analysis. Be critical and thorough.",
            llm_config=llm_config,
        )

        writer = autogen.AssistantAgent(
            name="Writer",
            system_message="You are a technical writer. Your job is to produce the "
                           "final, polished answer based on the analysis provided. "
                           "Make sure the answer is complete, accurate, and well-structured.",
            llm_config=llm_config,
        )

        user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
            system_message=f"A task has been assigned: {prompt}\n\n"
                           "Coordinate with the Researcher, Analyst, and Writer "
                           "to solve this task. Start by asking the Researcher to begin.",
        )

        # Create group chat
        groupchat = autogen.GroupChat(
            agents=[user_proxy, researcher, analyst, writer],
            messages=[],
            max_round=12,
            speaker_selection_method="auto",
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # Initiate conversation
        user_proxy.initiate_chat(
            manager,
            message=f"Please solve the following task by coordinating with your team:\n\n{prompt}",
        )

        # Extract final answer from conversation
        messages = groupchat.messages
        final_output = self._extract_final_answer(messages)
        return final_output

    def _build_prompt(self, task_input: TaskInput) -> str:
        parts = [f"Task: {task_input.description}"]
        if task_input.context:
            parts.append(f"\nContext:\n{task_input.context}")
        if task_input.input_data:
            import json
            parts.append(f"\nInput Data:\n```json\n{json.dumps(task_input.input_data, indent=2)}\n```")
        return "\n".join(parts)

    def _extract_final_answer(self, messages: list) -> str:
        """Extract the final answer from the conversation history."""
        # Look for the last message from the Writer or Analyst
        for msg in reversed(messages):
            content = str(msg.get("content", ""))
            if content and len(content) > 50:
                return content
        # Fallback: return last message
        if messages:
            return str(messages[-1].get("content", ""))
        return "No output produced."

    def agent_names(self) -> list[str]:
        return ["UserProxy", "Researcher", "Analyst", "Writer"]

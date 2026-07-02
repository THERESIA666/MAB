"""Base adapter interface and instrumented LLM for multi-agent frameworks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time
import os

from benchmark_toolkit.core.task import TaskInput
from benchmark_toolkit.core.metrics import RunMetrics, LLMCallRecord


@dataclass
class AdapterConfig:
    """Configuration for a framework adapter."""
    adapter_name: str
    llm_model: str = "deepseek-v4-flash"
    llm_provider: str = "deepseek"  # anthropic, openai, deepseek
    api_key: Optional[str] = None
    api_base: Optional[str] = None   # custom API base URL
    temperature: float = 0.0
    max_tokens: int = 4096
    thinking_mode: bool = False      # enable thinking/reasoning mode (DeepSeek)
    extra: dict = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        return os.environ.get(env_map.get(self.llm_provider, ""), "")

    def resolve_api_base(self) -> Optional[str]:
        if self.api_base:
            return self.api_base
        if self.llm_provider == "deepseek":
            return os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        if self.llm_provider == "openai":
            return os.environ.get("OPENAI_API_BASE", None)
        return None


class InstrumentedLLM:
    """A wrapper around LLM clients that records all calls for benchmarking.

    Supports Anthropic, OpenAI, and DeepSeek backends transparently.
    """

    # Pricing per 1M tokens (input, output) in USD — updated June 2026
    PRICING = {
        # DeepSeek V4 models (current)
        "deepseek-v4-flash": (0.14, 0.28),
        "deepseek-v4-pro": (0.435, 0.87),        # promo pricing
        "deepseek-v4-pro-standard": (1.74, 3.48),  # standard pricing
        # DeepSeek legacy (aliases → v4-flash, retire July 24 2026)
        "deepseek-chat": (0.14, 0.28),
        "deepseek-reasoner": (0.14, 0.28),
        # Anthropic models
        "claude-sonnet-4-6-20251001": (3.0, 15.0),
        "claude-opus-4-8-20250514": (15.0, 75.0),
        "claude-haiku-4-5-20251001": (0.80, 4.0),
        # OpenAI models
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
    }

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.call_log: list[LLMCallRecord] = []
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = self.config.resolve_api_key()
            api_base = self.config.resolve_api_base()

            if self.config.llm_provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            elif self.config.llm_provider == "deepseek":
                import openai
                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url=api_base or "https://api.deepseek.com",
                )
            else:
                import openai
                kwargs = {"api_key": api_key}
                if api_base:
                    kwargs["base_url"] = api_base
                self._client = openai.OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: list[dict],
        agent_name: str = "unknown",
        purpose: str = "general",
        **kwargs,
    ) -> str:
        """Send a chat completion request and record metrics."""
        start = time.time()

        model = kwargs.pop("model", self.config.llm_model)
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        temperature = kwargs.pop("temperature", self.config.temperature)

        if self.config.llm_provider == "anthropic":
            import anthropic
            system_msg = ""
            user_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_messages.append(m)
            response = self.client.messages.create(
                model=model,
                system=system_msg if system_msg else anthropic.NOT_GIVEN,
                messages=user_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            content = response.content[0].text
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
        else:
            # OpenAI-compatible API (works for OpenAI, DeepSeek, etc.)
            extra_body = {}
            if self.config.llm_provider == "deepseek" and self.config.thinking_mode:
                extra_body["thinking"] = {"type": "enabled"}

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

        elapsed = (time.time() - start) * 1000

        record = LLMCallRecord(
            timestamp=start,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=elapsed,
            agent_name=agent_name,
            purpose=purpose,
        )
        self.call_log.append(record)

        return content

    def reset_log(self) -> None:
        self.call_log.clear()

    def aggregate_metrics(self) -> RunMetrics:
        """Compute RunMetrics from the accumulated call log."""
        total_prompt = sum(r.prompt_tokens for r in self.call_log)
        total_completion = sum(r.completion_tokens for r in self.call_log)
        total_time = sum(r.duration_ms for r in self.call_log)

        if self.call_log:
            wall_start = min(r.timestamp for r in self.call_log)
            wall_end = max(r.timestamp + r.duration_ms / 1000 for r in self.call_log)
            wall_time = (wall_end - wall_start) * 1000
            overhead = max(0, wall_time - total_time)
        else:
            overhead = 0.0

        # Cost estimation
        price_input, price_output = self.PRICING.get(
            self.config.llm_model, (1.0, 5.0)
        )
        cost = (total_prompt / 1_000_000) * price_input + \
               (total_completion / 1_000_000) * price_output

        return RunMetrics(
            total_tokens=total_prompt + total_completion,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_time_ms=total_time,
            estimated_cost_usd=cost,
            call_log=list(self.call_log),
            orchestration_overhead_ms=overhead,
        )


class BaseAdapter(ABC):
    """Abstract interface for multi-agent framework adapters."""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.llm = InstrumentedLLM(config)

    @property
    def name(self) -> str:
        return self.config.adapter_name

    @abstractmethod
    def solve(self, task_input: TaskInput) -> str:
        """Execute the task using the multi-agent framework."""
        ...

    def run(self, task_input: TaskInput) -> tuple[str, RunMetrics]:
        """Run a task and return (output, metrics)."""
        self.llm.reset_log()
        start = time.time()

        try:
            output = self.solve(task_input)
        except Exception as e:
            output = f"ERROR: {str(e)}"

        elapsed = (time.time() - start) * 1000
        metrics = self.llm.aggregate_metrics()
        metrics.total_time_ms = elapsed

        llm_time = sum(r.duration_ms for r in self.llm.call_log)
        metrics.orchestration_overhead_ms = max(0, elapsed - llm_time)

        return output, metrics

    @abstractmethod
    def agent_names(self) -> list[str]:
        """Return the names of agents this adapter will deploy."""
        ...

    def cleanup(self) -> None:
        """Optional cleanup after a run."""
        pass

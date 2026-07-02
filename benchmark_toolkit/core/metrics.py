"""Metrics collection and reporting for multi-agent benchmark runs."""

from dataclasses import dataclass, field
from typing import Optional
import time
import json


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    agent_name: str = "unknown"
    purpose: str = "unknown"  # e.g., "reasoning", "tool_call", "summarization"


@dataclass
class RunMetrics:
    """Aggregated metrics from a single benchmark run."""
    # Core metrics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_ms: float = 0.0

    # Agent interaction metrics
    agent_interactions: int = 0       # how many times agents communicated
    tool_calls: int = 0               # how many tool invocations
    agent_count: int = 1              # how many agents participated

    # Cost estimation (USD, approximate)
    estimated_cost_usd: float = 0.0

    # Detailed call log (for debugging/analysis)
    call_log: list = field(default_factory=list)

    # Overhead analysis
    orchestration_overhead_ms: float = 0.0  # time spent not in LLM calls
    internal_roundtrips: int = 0            # rounds of agent interaction

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_time_ms": self.total_time_ms,
            "agent_interactions": self.agent_interactions,
            "tool_calls": self.tool_calls,
            "agent_count": self.agent_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "orchestration_overhead_ms": self.orchestration_overhead_ms,
            "internal_roundtrips": self.internal_roundtrips,
            "llm_calls": len(self.call_log),
        }

    @property
    def efficiency_ratio(self) -> float:
        """Ratio of completion tokens to total tokens — higher means less overhead."""
        if self.total_tokens == 0:
            return 0.0
        return self.completion_tokens / self.total_tokens

    @property
    def tokens_per_interaction(self) -> float:
        """Average tokens consumed per agent interaction."""
        if self.agent_interactions == 0:
            return float(self.total_tokens)
        return self.total_tokens / self.agent_interactions


@dataclass
class TaskResult:
    """Result of running a single task with a single adapter."""
    task_id: str
    task_name: str
    adapter_name: str
    success: bool
    score: float  # 0.0 - 1.0
    metrics: RunMetrics
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all tasks and adapters."""
    benchmark_id: str
    timestamp: str
    adapters: list[str]
    tasks: list[str]
    results: list[TaskResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_result(self, adapter_name: str, task_id: str) -> Optional[TaskResult]:
        for r in self.results:
            if r.adapter_name == adapter_name and r.task_id == task_id:
                return r
        return None

    def get_adapter_summary(self, adapter_name: str) -> dict:
        """Get aggregate stats for one adapter across all tasks."""
        adapter_results = [r for r in self.results if r.adapter_name == adapter_name]
        if not adapter_results:
            return {}
        n = len(adapter_results)
        return {
            "adapter": adapter_name,
            "tasks_completed": n,
            "avg_score": sum(r.score for r in adapter_results) / n,
            "success_rate": sum(1 for r in adapter_results if r.success) / n,
            "total_tokens": sum(r.metrics.total_tokens for r in adapter_results),
            "total_cost_usd": sum(r.metrics.estimated_cost_usd for r in adapter_results),
            "avg_time_ms": sum(r.metrics.total_time_ms for r in adapter_results) / n,
            "avg_efficiency": sum(r.metrics.efficiency_ratio for r in adapter_results) / n,
            "avg_interactions": sum(r.metrics.agent_interactions for r in adapter_results) / n,
        }

    def get_ranking(self, metric: str = "avg_score") -> list[tuple[str, float]]:
        """Rank adapters by a specific summary metric."""
        summaries = [(a, self.get_adapter_summary(a)) for a in self.adapters]
        valid = [(a, s[metric]) for a, s in summaries if s and metric in s]
        return sorted(valid, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp,
            "adapters": self.adapters,
            "tasks": self.tasks,
            "results": [
                {
                    "task_id": r.task_id,
                    "task_name": r.task_name,
                    "adapter_name": r.adapter_name,
                    "success": r.success,
                    "score": r.score,
                    "metrics": r.metrics.to_dict(),
                    "output": r.output,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkReport":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = cls(
            benchmark_id=data["benchmark_id"],
            timestamp=data["timestamp"],
            adapters=data["adapters"],
            tasks=data["tasks"],
            metadata=data.get("metadata", {}),
        )
        for r in data["results"]:
            m = r["metrics"]
            metrics = RunMetrics(
                total_tokens=m["total_tokens"],
                prompt_tokens=m["prompt_tokens"],
                completion_tokens=m["completion_tokens"],
                total_time_ms=m["total_time_ms"],
                agent_interactions=m.get("agent_interactions", 0),
                tool_calls=m.get("tool_calls", 0),
                agent_count=m.get("agent_count", 1),
                estimated_cost_usd=m.get("estimated_cost_usd", 0.0),
                orchestration_overhead_ms=m.get("orchestration_overhead_ms", 0.0),
                internal_roundtrips=m.get("internal_roundtrips", 0),
            )
            report.results.append(TaskResult(
                task_id=r["task_id"],
                task_name=r["task_name"],
                adapter_name=r["adapter_name"],
                success=r["success"],
                score=r["score"],
                metrics=metrics,
                output=r.get("output"),
                error=r.get("error"),
                duration_ms=r.get("duration_ms", 0.0),
            ))
        return report

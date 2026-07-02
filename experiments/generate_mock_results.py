"""Generate simulated benchmark results for testing the dashboard.

This creates realistic-looking mock data without making actual API calls,
useful for developing and testing the visualization dashboard.
"""

import json
import random
import os
from datetime import datetime, timezone

# Set seed for reproducibility
random.seed(42)

ADAPTERS = ["single-agent", "crewai", "autogen", "langgraph"]
TASKS = ["mhq-1", "mhq-2", "mhq-3", "code-1", "code-2", "data-1",
         "research-1", "research-2", "plan-1"]
TASK_NAMES = {
    "mhq-1": "Multi-hop QA: Official language of AlphaGo HQ country",
    "mhq-2": "Multi-hop QA: Birth year of first Moon walker",
    "mhq-3": "Multi-hop QA: Chemical symbol of Einstein's element",
    "code-1": "Code: Palindrome function",
    "code-2": "Code: Fibonacci function",
    "data-1": "Data Analysis: Online Store Sales",
    "research-1": "Research: Transformer vs Mamba architectures",
    "research-2": "Research: Speculative decoding in LLMs",
    "plan-1": "Planning: Tech Conference Organization",
}

# Simulated performance characteristics per adapter
# (single-agent is fast but less thorough, multi-agent is slower but better)
ADAPTER_PROFILES = {
    "single-agent": {
        "score_range": (0.45, 0.75),
        "success_rate": 0.55,
        "tokens_per_task": (800, 2500),
        "time_per_task_ms": (3000, 15000),
        "agent_count": 1,
        "interactions": (1, 1),
    },
    "crewai": {
        "score_range": (0.55, 0.88),
        "success_rate": 0.70,
        "tokens_per_task": (3000, 8000),
        "time_per_task_ms": (15000, 45000),
        "agent_count": 3,
        "interactions": (3, 8),
    },
    "autogen": {
        "score_range": (0.50, 0.85),
        "success_rate": 0.68,
        "tokens_per_task": (3500, 9000),
        "time_per_task_ms": (20000, 50000),
        "agent_count": 4,
        "interactions": (4, 12),
    },
    "langgraph": {
        "score_range": (0.55, 0.90),
        "success_rate": 0.72,
        "tokens_per_task": (2500, 7000),
        "time_per_task_ms": (12000, 35000),
        "agent_count": 3,
        "interactions": (3, 6),
    },
}


def generate():
    results = []

    for task_id in TASKS:
        for adapter_name in ADAPTERS:
            profile = ADAPTER_PROFILES[adapter_name]

            # Generate metrics
            prompt_tokens = random.randint(
                int(profile["tokens_per_task"][0] * 0.6),
                int(profile["tokens_per_task"][1] * 0.6),
            )
            completion_tokens = random.randint(
                int(profile["tokens_per_task"][0] * 0.4),
                int(profile["tokens_per_task"][1] * 0.4),
            )
            total_time = random.randint(
                profile["time_per_task_ms"][0],
                profile["time_per_task_ms"][1],
            )
            llm_ratio = 0.7 if adapter_name == "single-agent" else 0.55
            llm_time = total_time * llm_ratio
            overhead = total_time * (1 - llm_ratio)

            score = round(random.uniform(*profile["score_range"]), 3)
            success = score >= 0.5

            # Cost estimation
            prompt_cost = (prompt_tokens / 1_000_000) * 3.0  # $3/1M tokens input
            completion_cost = (completion_tokens / 1_000_000) * 15.0  # $15/1M tokens output
            cost = round(prompt_cost + completion_cost, 6)

            interactions = random.randint(*profile["interactions"])

            result = {
                "task_id": task_id,
                "task_name": TASK_NAMES[task_id],
                "adapter_name": adapter_name,
                "success": success,
                "score": score,
                "output": f"[Simulated output for {task_id} by {adapter_name}]",
                "error": None,
                "duration_ms": total_time,
                "metrics": {
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_time_ms": total_time,
                    "agent_interactions": interactions,
                    "tool_calls": 0 if task_id.startswith("mhq") else random.randint(1, 4),
                    "agent_count": profile["agent_count"],
                    "estimated_cost_usd": cost,
                    "orchestration_overhead_ms": overhead,
                    "internal_roundtrips": interactions,
                    "llm_calls": interactions,
                },
            }
            results.append(result)

    report = {
        "benchmark_id": "mock-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapters": ADAPTERS,
        "tasks": TASKS,
        "results": results,
        "metadata": {
            "repeat": 1,
            "task_count": len(TASKS),
            "adapter_count": len(ADAPTERS),
            "note": "MOCK DATA — for dashboard testing only. Not real benchmark results.",
        },
    }

    os.makedirs("experiments/results", exist_ok=True)
    output_path = "experiments/results/mock_benchmark.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Mock benchmark data generated: {output_path}")
    print(f"   Tasks: {len(TASKS)} | Adapters: {len(ADAPTERS)} | Results: {len(results)}")
    print()
    print("Launch dashboard with:")
    print("  streamlit run benchmark_toolkit/dashboard/app.py")


if __name__ == "__main__":
    generate()

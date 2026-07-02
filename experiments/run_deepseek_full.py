"""Run full benchmark experiment with DeepSeek — single vs multi-step comparison.

Runs all 9 tasks against both single-agent and multi-step collaboration
adapters, producing comparative results for the paper.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.adapters.single_agent import SingleAgentAdapter
from benchmark_toolkit.adapters.multi_step_adapter import MultiStepAdapter
from benchmark_toolkit.core.adapter import AdapterConfig
from benchmark_toolkit.core.runner import BenchmarkRunner

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def main():
    print("=" * 60)
    print("Multi-Agent Benchmark — Single vs Multi-Step (DeepSeek)")
    print("=" * 60)

    suite = create_default_suite()
    print(f"Tasks: {len(suite)}")

    # Single-agent adapter
    sa_config = AdapterConfig(
        adapter_name="single-agent",
        llm_model="deepseek-chat",
        llm_provider="deepseek",
        api_key=API_KEY,
        temperature=0.0,
        max_tokens=4096,
    )
    single_agent = SingleAgentAdapter(sa_config)

    # Multi-step collaboration adapter
    ms_config = AdapterConfig(
        adapter_name="multi-step",
        llm_model="deepseek-chat",
        llm_provider="deepseek",
        api_key=API_KEY,
        temperature=0.0,
        max_tokens=4096,
    )
    multi_step = MultiStepAdapter(ms_config)

    adapters = [single_agent, multi_step]
    print(f"Adapters: {len(adapters)} ({', '.join(a.name for a in adapters)})")
    print()

    runner = BenchmarkRunner(
        adapters=adapters,
        repeat=1,
        timeout_seconds=300,
        verbose=True,
    )

    start = time.time()
    report = runner.run(suite)
    elapsed = time.time() - start

    output_path = "experiments/results/deepseek_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.to_json(output_path)

    print()
    print("=" * 60)
    print(f"Experiment completed in {elapsed:.1f}s")
    print(f"Report saved to: {output_path}")

    # Detailed comparison
    print()
    print("=" * 60)
    print("COMPARISON: Single-Agent vs Multi-Step Collaboration")
    print("=" * 60)

    for adapter_name in ["single-agent", "multi-step"]:
        summary = report.get_adapter_summary(adapter_name)
        if summary:
            print(f"\n--- {adapter_name} ---")
            print(f"  Avg Score:     {summary['avg_score']:.3f}")
            print(f"  Success Rate:  {summary['success_rate']:.0%}")
            print(f"  Total Tokens:  {summary['total_tokens']:,}")
            print(f"  Total Cost:    ${summary['total_cost_usd']:.4f}")
            print(f"  Avg Time:      {summary['avg_time_ms']/1000:.1f}s")
            print(f"  Avg Efficiency: {summary['avg_efficiency']:.3f}")
            print(f"  Avg Interactions: {summary['avg_interactions']:.1f}")

    # Per-task comparison
    print()
    print("=" * 60)
    print("PER-TASK SCORE COMPARISON")
    print("=" * 60)
    print(f"{'Task':<12} {'Single':>8} {'Multi-Step':>12} {'Delta':>8}")
    print("-" * 42)
    for task_id in suite.task_ids:
        sa = report.get_result("single-agent", task_id)
        ms = report.get_result("multi-step", task_id)
        sa_score = sa.score if sa else 0
        ms_score = ms.score if ms else 0
        delta = ms_score - sa_score
        sign = "+" if delta > 0 else ""
        print(f"{task_id:<12} {sa_score:>8.2f} {ms_score:>12.2f} {sign}{delta:>7.2f}")

    # Average delta
    deltas = []
    for task_id in suite.task_ids:
        sa = report.get_result("single-agent", task_id)
        ms = report.get_result("multi-step", task_id)
        if sa and ms:
            deltas.append(ms.score - sa.score)
    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        print("-" * 42)
        print(f"{'AVERAGE':<12} {sum(sa.score for task_id in suite.task_ids if (sa := report.get_result('single-agent', task_id)))/len(deltas):>8.2f} {sum(report.get_result('multi-step', task_id).score for task_id in suite.task_ids if (ms := report.get_result('multi-step', task_id)))/len(deltas):>12.2f} {'+' if avg_delta > 0 else ''}{avg_delta:>7.2f}")


if __name__ == "__main__":
    main()

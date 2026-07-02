"""Run the benchmark experiment with real API calls.

This script:
1. Sets up the single-agent adapter (no framework dependency needed)
2. Runs all benchmark tasks
3. Saves results to experiments/results/

For multi-agent frameworks, install the optional dependencies:
    pip install multi-agent-benchmark[crewai]
    pip install multi-agent-benchmark[autogen]
    pip install multi-agent-benchmark[langgraph]

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --model claude-haiku-4-5-20251001
    python experiments/run_experiment.py --adapter single-agent --adapter crewai
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.adapters.single_agent import SingleAgentAdapter
from benchmark_toolkit.core.adapter import AdapterConfig, BaseAdapter
from benchmark_toolkit.core.runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Multi-Agent Benchmark Experiment"
    )
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6-20251001",
                        help="LLM model to use")
    parser.add_argument("--provider", "-p", default="anthropic",
                        help="LLM provider (anthropic or openai)")
    parser.add_argument("--adapter", "-a", action="append", default=None,
                        help="Adapter to use (can repeat). Default: single-agent")
    parser.add_argument("--task", "-t", action="append", default=None,
                        help="Specific task IDs (can repeat). Default: all")
    parser.add_argument("--repeat", "-r", type=int, default=1,
                        help="Repetitions per task")
    parser.add_argument("--output", "-o", default="experiments/results/benchmark_result.json")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def create_adapters(adapter_names, model, provider):
    """Create adapter instances."""
    if adapter_names is None:
        adapter_names = ["single-agent"]

    adapters = []
    for name in adapter_names:
        config = AdapterConfig(
            adapter_name=name,
            llm_model=model,
            llm_provider=provider,
        )
        if name == "single-agent":
            adapters.append(SingleAgentAdapter(config))
        elif name == "crewai":
            try:
                from benchmark_toolkit.adapters.crewai_adapter import CrewAIAdapter
                adapters.append(CrewAIAdapter(config))
            except ImportError as e:
                print(f"⚠️  Skipping crewai: {e}")
        elif name == "autogen":
            try:
                from benchmark_toolkit.adapters.autogen_adapter import AutoGenAdapter
                adapters.append(AutoGenAdapter(config))
            except ImportError as e:
                print(f"⚠️  Skipping autogen: {e}")
        elif name == "langgraph":
            try:
                from benchmark_toolkit.adapters.langgraph_adapter import LangGraphAdapter
                adapters.append(LangGraphAdapter(config))
            except ImportError as e:
                print(f"⚠️  Skipping langgraph: {e}")
        else:
            print(f"⚠️  Unknown adapter: {name}")

    return adapters


def main():
    args = parse_args()

    print("=" * 60)
    print("🤖 Multi-Agent Collaboration Benchmark Experiment")
    print("=" * 60)
    print(f"Model: {args.model} ({args.provider})")
    print(f"Repeat: {args.repeat}x per task")
    print()

    # Check API key
    if args.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("❌ ANTHROPIC_API_KEY environment variable not set.")
            print("   Set it with: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
            print("   Or generate mock data: python experiments/generate_mock_results.py")
            sys.exit(1)
    elif args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable not set.")
            print("   Set it with: $env:OPENAI_API_KEY = 'sk-...'")
            print("   Or generate mock data: python experiments/generate_mock_results.py")
            sys.exit(1)

    # Create adapters
    adapters = create_adapters(args.adapter, args.model, args.provider)
    if not adapters:
        print("❌ No adapters available.")
        sys.exit(1)
    print(f"🔌 Adapters: {', '.join(a.name for a in adapters)}")

    # Create task suite
    suite = create_default_suite()
    if args.task:
        from benchmark_toolkit.core.task import TaskSuite
        selected = [t for t in suite if t.task_id in args.task]
        suite = TaskSuite(selected)
    print(f"📋 Tasks: {len(suite)}")
    print()

    # Run
    runner = BenchmarkRunner(
        adapters=adapters,
        repeat=args.repeat,
        timeout_seconds=args.timeout,
        verbose=not args.quiet,
    )

    start = time.time()
    report = runner.run(suite)
    elapsed = time.time() - start

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report.to_json(args.output)

    print()
    print("=" * 60)
    print(f"✅ Experiment completed in {elapsed:.1f}s")
    print(f"📊 Report saved to: {args.output}")

    # Summary
    print()
    print("🏆 Final Rankings (by average score):")
    for i, (name, score) in enumerate(report.get_ranking("avg_score"), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        summary = report.get_adapter_summary(name)
        print(f"  {emoji} {name}: score={score:.3f}, "
              f"success_rate={summary.get('success_rate', 0):.0%}, "
              f"tokens={summary.get('total_tokens', 0):,}, "
              f"cost=${summary.get('total_cost_usd', 0):.4f}")

    print()
    print("📈 View results: streamlit run benchmark_toolkit/dashboard/app.py")


if __name__ == "__main__":
    main()

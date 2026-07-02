"""Multi-model comparison experiment.

Compares DeepSeek V4 Flash vs V4 Pro across single-agent and multi-step
collaboration strategies, including thinking/reasoning mode for Pro.

Experiment matrix:
  v4-flash  × single-agent  → baseline (fast & cheap)
  v4-flash  × multi-step    → collaboration on budget
  v4-pro    × single-agent  → powerful solo
  v4-pro    × multi-step    → powerful collaboration
  v4-pro    × single-agent + thinking  → reasoning baseline
  v4-pro    × multi-step + thinking    → reasoning collaboration
"""

import sys, os, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.adapters.single_agent import SingleAgentAdapter
from benchmark_toolkit.adapters.multi_step_adapter import MultiStepAdapter
from benchmark_toolkit.core.adapter import AdapterConfig
from benchmark_toolkit.core.runner import BenchmarkRunner
from benchmark_toolkit.core.metrics import BenchmarkReport

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Tasks that benefit from deeper reasoning
HARD_TASK_IDS = {"code-1", "code-2", "research-1", "research-2", "plan-1"}


def create_config(name, model, thinking=False):
    return AdapterConfig(
        adapter_name=name,
        llm_model=model,
        llm_provider="deepseek",
        api_key=API_KEY,
        temperature=0.0,
        max_tokens=4096,
        thinking_mode=thinking,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run all 6 configs including thinking mode")
    parser.add_argument("--output", "-o", default="experiments/results/multimodel_comparison.json")
    args = parser.parse_args()

    suite = create_default_suite()

    # Base configurations (always run)
    configs = [
        ("v4-flash-single", "deepseek-v4-flash", False, SingleAgentAdapter),
        ("v4-flash-multi", "deepseek-v4-flash", False, MultiStepAdapter),
        ("v4-pro-single", "deepseek-v4-pro", False, SingleAgentAdapter),
        ("v4-pro-multi", "deepseek-v4-pro", False, MultiStepAdapter),
    ]

    if args.full:
        configs += [
            ("v4-pro-think-single", "deepseek-v4-pro", True, SingleAgentAdapter),
            ("v4-pro-think-multi", "deepseek-v4-pro", True, MultiStepAdapter),
        ]

    print("=" * 70)
    print("Multi-Model Benchmark: DeepSeek V4 Flash vs V4 Pro")
    print("=" * 70)
    print(f"Configurations: {len(configs)}")
    if args.full:
        print("  (including thinking/reasoning mode)")
    print(f"Tasks: {len(suite)}")
    print(f"Total runs: {len(configs) * len(suite)}")
    print()

    # Build adapters
    adapters = []
    for name, model, thinking, adapter_cls in configs:
        cfg = create_config(name, model, thinking)
        adapter = adapter_cls(cfg)
        adapters.append(adapter)
        think_label = " [thinking]" if thinking else ""
        print(f"  {name}: {model}{think_label} ({adapter_cls.__name__})")

    print()

    # For thinking mode, filter to hard tasks only
    if args.full:
        print("Note: thinking mode runs on hard tasks only (code, research, planning)")
        hard_tasks = [t for t in suite if t.task_id in HARD_TASK_IDS]
        from benchmark_toolkit.core.task import TaskSuite
        think_suite = TaskSuite(hard_tasks)
    else:
        think_suite = suite

    # Run non-thinking configs on full suite, thinking configs on hard tasks
    all_results = []

    for adapter in adapters:
        is_thinking = adapter.config.thinking_mode
        target_suite = think_suite if is_thinking else suite

        print(f"\n{'='*50}")
        print(f"Running: {adapter.name}")
        print(f"  Model: {adapter.config.llm_model}")
        print(f"  Thinking: {adapter.config.thinking_mode}")
        print(f"  Tasks: {len(target_suite)}")
        print(f"{'='*50}")

        runner = BenchmarkRunner(
            adapters=[adapter],
            repeat=1,
            timeout_seconds=300,
            verbose=True,
        )
        report = runner.run(target_suite)
        all_results.extend(report.results)

    # Merge all results into a single report
    final_report = BenchmarkReport(
        benchmark_id="multimodel-v4",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        adapters=[a.name for a in adapters],
        tasks=sorted(set(r.task_id for r in all_results)),
        results=all_results,
        metadata={
            "description": "DeepSeek V4 Flash vs V4 Pro comparison",
            "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
            "thinking_mode_included": args.full,
        },
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_report.to_json(args.output)

    # Print comprehensive comparison
    print("\n" + "=" * 70)
    print("RESULTS: Multi-Model Comparison")
    print("=" * 70)

    _print_summary_table(final_report)
    _print_model_comparison(final_report)
    _print_task_matrix(final_report)


def _print_summary_table(report: BenchmarkReport):
    """Print aggregate summary for all adapters."""
    print("\n--- Aggregate Summary ---")
    header = f"{'Adapter':<25} {'Score':>6} {'Success':>8} {'Tokens':>9} {'Time(s)':>8} {'Cost(USD)':>10}"
    print(header)
    print("-" * len(header))

    for name in report.adapters:
        s = report.get_adapter_summary(name)
        if s:
            print(f"{name:<25} {s['avg_score']:>6.3f} {s['success_rate']:>7.0%} "
                  f"{s['total_tokens']:>9,} {s['avg_time_ms']/1000:>8.1f} "
                  f"${s['total_cost_usd']:>9.4f}")


def _print_model_comparison(report: BenchmarkReport):
    """Compare model-level aggregates."""
    print("\n--- Model-Level Comparison (averaged over strategies) ---")

    models = {
        "v4-flash": ["v4-flash-single", "v4-flash-multi"],
        "v4-pro": ["v4-pro-single", "v4-pro-multi"],
    }

    header = f"{'Model':<10} {'Avg Score':>10} {'Avg Tokens':>12} {'Avg Time':>10} {'Avg Cost':>10}"
    print(header)
    print("-" * len(header))

    for model_name, adapter_names in models.items():
        score_sum = token_sum = time_sum = cost_sum = count = 0
        task_set = set()
        for name in adapter_names:
            for r in report.results:
                if r.adapter_name == name and r.task_id in task_set or True:
                    pass
            results = [r for r in report.results if r.adapter_name == name]
            for r in results:
                score_sum += r.score
                token_sum += r.metrics.total_tokens
                time_sum += r.metrics.total_time_ms
                cost_sum += r.metrics.estimated_cost_usd
                count += 1
        if count:
            print(f"{model_name:<10} {score_sum/count:>10.3f} {token_sum:>12,} "
                  f"{time_sum/count/1000:>10.1f} ${cost_sum:>9.4f}")


def _print_task_matrix(report: BenchmarkReport):
    """Print per-task comparison matrix."""
    print("\n--- Per-Task Score Matrix ---")
    tasks = sorted(set(r.task_id for r in report.results))
    adapters = report.adapters

    # Header
    header = f"{'Task':<12}"
    for a in adapters:
        header += f" {a[:18]:>18}"
    print(header)
    print("-" * len(header))

    for task_id in tasks:
        row = f"{task_id:<12}"
        for adapter_name in adapters:
            result = report.get_result(adapter_name, task_id)
            if result:
                score = result.score
                marker = "*" if result.success else " "
                row += f" {score:>17.2f}{marker}"
            else:
                row += f" {'N/A':>18}"
        print(row)

    # Best per task
    print("\n--- Best Configuration Per Task ---")
    for task_id in tasks:
        best_score = -1
        best_name = ""
        for adapter_name in adapters:
            result = report.get_result(adapter_name, task_id)
            if result and result.score > best_score:
                best_score = result.score
                best_name = adapter_name
        print(f"  {task_id:<12} → {best_name} ({best_score:.2f})")


if __name__ == "__main__":
    main()

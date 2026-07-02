"""CLI entry point for the Multi-Agent Benchmark Toolkit.

Usage:
    mab run                          # Run full benchmark
    mab run --adapter single-agent   # Run with specific adapter
    mab run --task mhq-1             # Run specific task
    mab list                         # List available tasks and adapters
    mab dashboard                    # Launch Streamlit dashboard
"""

import click
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_toolkit.tasks.task_registry import create_default_suite
from benchmark_toolkit.adapters.single_agent import SingleAgentAdapter
from benchmark_toolkit.core.adapter import AdapterConfig, BaseAdapter
from benchmark_toolkit.core.runner import BenchmarkRunner
from benchmark_toolkit.core.metrics import BenchmarkReport


@click.group()
@click.version_option(version="0.1.0", prog_name="mab")
def cli():
    """Multi-Agent Collaboration Benchmark Toolkit.

    Compare multi-agent frameworks (CrewAI, AutoGen, LangGraph) against
    a single-agent baseline on standardized collaboration tasks.
    """
    pass


@cli.command("run")
@click.option("--adapter", "-a", multiple=True,
              help="Adapter to use (can repeat). Default: single-agent only.")
@click.option("--task", "-t", multiple=True,
              help="Specific task IDs to run. Default: all tasks.")
@click.option("--repeat", "-r", default=1, type=int,
              help="Number of times to repeat each task.")
@click.option("--output", "-o", default="experiments/results/benchmark_result.json",
              help="Output JSON file path.")
@click.option("--model", "-m", default="deepseek-v4-flash",
              help="LLM model to use.")
@click.option("--provider", "-p", default="anthropic",
              help="LLM provider (anthropic or openai).")
@click.option("--timeout", type=int, default=300,
              help="Timeout per task in seconds.")
@click.option("--quiet", is_flag=True, help="Suppress progress output.")
def run_benchmark(adapter, task, repeat, output, model, provider, timeout, quiet):
    """Run benchmark tasks against multi-agent frameworks."""
    click.echo("=== Multi-Agent Collaboration Benchmark ===")
    click.echo(f"   Model: {model} ({provider})")
    click.echo()

    # Create task suite
    suite = create_default_suite()
    if task:
        from benchmark_toolkit.core.task import TaskSuite
        selected = [t for t in suite if t.task_id in task]
        if not selected:
            click.echo(f"[ERROR] No matching tasks found for: {task}", err=True)
            click.echo(f"   Available: {suite.task_ids}")
            sys.exit(1)
        suite = TaskSuite(selected)

    click.echo(f"Tasks: {len(suite)} ({', '.join(suite.task_ids)})")

    # Create adapters
    adapters = _create_adapters(adapter, model, provider)
    click.echo(f"Adapters: {len(adapters)} ({', '.join(a.name for a in adapters)})")
    click.echo()

    # Create output directory
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Run benchmark
    runner = BenchmarkRunner(
        adapters=adapters,
        repeat=repeat,
        timeout_seconds=timeout,
        verbose=not quiet,
    )

    report = runner.run(suite)

    # Save report
    report.to_json(output)
    click.echo(f"\n[OK] Report saved to: {output}")

    # Quick summary
    click.echo()
    click.echo("=" * 60)
    click.echo("Rankings (by average score):")
    rankings = report.get_ranking("avg_score")
    for i, (name, score) in enumerate(rankings, 1):
        prefix = "[1st]" if i == 1 else "[2nd]" if i == 2 else "[3rd]" if i == 3 else f"  {i}."
        click.echo(f"  {prefix} {name}: {score:.3f}")


@cli.command("list")
def list_resources():
    """List available tasks and supported adapters."""
    suite = create_default_suite()

    click.echo("Available Tasks:\n")
    for task in suite:
        click.echo(f"  [{task.task_id}] {task.name}")
        click.echo(f"      Category: {task.category} | Difficulty: {task.difficulty} | "
                    f"Min agents: {task.min_agents_recommended}")
        click.echo(f"      Tags: {', '.join(task.tags)}")
        click.echo()

    click.echo("\nAvailable Adapters:")
    click.echo("  * single-agent     - Single LLM call baseline (no framework needed)")
    click.echo("  * crewai           - CrewAI multi-agent framework")
    click.echo("  * autogen          - Microsoft AutoGen framework")
    click.echo("  * langgraph        - LangChain LangGraph framework")


@cli.command("dashboard")
def launch_dashboard():
    """Launch the Streamlit visualization dashboard."""
    import subprocess

    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard", "app.py"
    )

    click.echo("Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", dashboard_path])


def _create_adapters(
    adapter_names: tuple[str, ...],
    model: str,
    provider: str,
) -> list:
    """Create adapter instances from names."""
    available = {
        "single-agent": lambda: SingleAgentAdapter(
            AdapterConfig(adapter_name="single-agent", llm_model=model,
                          llm_provider=provider)
        ),
        "crewai": lambda: _try_create(
            "crewai", model, provider
        ),
        "autogen": lambda: _try_create(
            "autogen", model, provider
        ),
        "langgraph": lambda: _try_create(
            "langgraph", model, provider
        ),
    }

    names = list(adapter_names) if adapter_names else ["single-agent"]
    adapters = []

    for name in names:
        if name not in available:
            click.echo(f"[WARNING] Unknown adapter: {name}. Skipping.", err=True)
            continue
        try:
            adapter = available[name]()
            adapters.append(adapter)
        except ImportError as e:
            click.echo(f"[WARNING] {e}. Skipping {name}.", err=True)

    if not adapters:
        click.echo("[ERROR] No adapters available. Install at least one framework.")
        sys.exit(1)

    return adapters


def _try_create(name: str, model: str, provider: str) -> BaseAdapter:
    """Try to create an adapter, with helpful error on missing dependency."""
    config = AdapterConfig(
        adapter_name=name,
        llm_model=model,
        llm_provider=provider,
    )
    if name == "crewai":
        from benchmark_toolkit.adapters.crewai_adapter import CrewAIAdapter
        return CrewAIAdapter(config)
    elif name == "autogen":
        from benchmark_toolkit.adapters.autogen_adapter import AutoGenAdapter
        return AutoGenAdapter(config)
    elif name == "langgraph":
        from benchmark_toolkit.adapters.langgraph_adapter import LangGraphAdapter
        return LangGraphAdapter(config)
    raise ValueError(f"Unknown adapter: {name}")


if __name__ == "__main__":
    cli()

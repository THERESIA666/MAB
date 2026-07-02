"""Benchmark runner — orchestrates tasks across adapters and collects results."""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from benchmark_toolkit.core.task import Task, TaskSuite, TaskInput
from benchmark_toolkit.core.adapter import BaseAdapter
from benchmark_toolkit.core.metrics import TaskResult, BenchmarkReport, RunMetrics

console = Console()


class BenchmarkRunner:
    """Runs benchmark tasks across multiple framework adapters.

    Usage:
        runner = BenchmarkRunner(adapters=[crewai_adapter, autogen_adapter])
        report = runner.run(task_suite)
        report.to_json("results.json")
    """

    def __init__(
        self,
        adapters: list[BaseAdapter],
        repeat: int = 1,
        timeout_seconds: int = 300,
        verbose: bool = True,
    ):
        if not adapters:
            raise ValueError("At least one adapter is required")
        self.adapters = adapters
        self.repeat = repeat
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

    def run(self, task_suite: TaskSuite) -> BenchmarkReport:
        """Run all tasks against all adapters."""
        report = BenchmarkReport(
            benchmark_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            adapters=[a.name for a in self.adapters],
            tasks=task_suite.task_ids,
            metadata={
                "repeat": self.repeat,
                "task_count": len(task_suite),
                "adapter_count": len(self.adapters),
            },
        )

        total_runs = len(task_suite) * len(self.adapters) * self.repeat

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Running benchmarks...", total=total_runs
            )

            for task in task_suite:
                for adapter in self.adapters:
                    for run_idx in range(self.repeat):
                        desc = f"[cyan]{adapter.name}[/cyan] on [yellow]{task.name}[/yellow]"
                        if self.repeat > 1:
                            desc += f" (run {run_idx + 1}/{self.repeat})"
                        progress.update(task_progress, description=desc)

                        result = self._run_single(task, adapter)
                        report.results.append(result)
                        progress.advance(task_progress)

        if self.verbose:
            self._print_summary(report)

        return report

    def _run_single(self, task: Task, adapter: BaseAdapter) -> TaskResult:
        """Run a single task with a single adapter."""
        task_input = task.generate_input()
        start = time.time()

        try:
            output, metrics = adapter.run(task_input)
            score = task.evaluate(output)
            success = score >= 0.5  # threshold for "success"
        except Exception as e:
            output = None
            metrics = RunMetrics()
            score = 0.0
            success = False
            error_msg = str(e)
        else:
            error_msg = None

        duration_ms = (time.time() - start) * 1000

        # Enrich metrics with agent interaction data
        metrics.agent_count = len(adapter.agent_names())
        metrics.internal_roundtrips = len(metrics.call_log)

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            adapter_name=adapter.name,
            success=success,
            score=score,
            metrics=metrics,
            output=output[:2000] if output else None,  # truncate for storage
            error=error_msg,
            duration_ms=duration_ms,
        )

    def _print_summary(self, report: BenchmarkReport) -> None:
        """Print a formatted summary table to the console."""
        console.print()
        console.print(f"[bold]Benchmark Report[/bold] ({report.benchmark_id})", style="bold cyan")
        console.print(f"Tasks: {len(report.tasks)} | Adapters: {len(report.adapters)} | "
                      f"Total runs: {len(report.results)}")
        console.print()

        # Per-adapter summary
        table = Table(title="Adapter Summary")
        table.add_column("Adapter", style="cyan")
        table.add_column("Avg Score", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Total Tokens", justify="right")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Cost (USD)", justify="right")

        for adapter_name in report.adapters:
            summary = report.get_adapter_summary(adapter_name)
            if summary:
                table.add_row(
                    adapter_name,
                    f"{summary['avg_score']:.2f}",
                    f"{summary['success_rate']:.0%}",
                    f"{summary['total_tokens']:,}",
                    f"{summary['avg_time_ms']/1000:.1f}",
                    f"${summary['total_cost_usd']:.4f}",
                )

        console.print(table)

        # Per-task comparison
        console.print()
        task_table = Table(title="Per-Task Score Comparison")
        task_table.add_column("Task", style="yellow")
        for adapter_name in report.adapters:
            task_table.add_column(adapter_name, justify="right")

        for task_id in report.tasks:
            row = [task_id]
            for adapter_name in report.adapters:
                result = report.get_result(adapter_name, task_id)
                if result:
                    style = "green" if result.success else "red"
                    row.append(f"[{style}]{result.score:.3f}[/{style}]")
                else:
                    row.append("N/A")
            task_table.add_row(*row)

        console.print(task_table)

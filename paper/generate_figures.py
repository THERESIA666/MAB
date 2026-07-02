"""Generate figures for the DAI 2026 paper from benchmark results.

Handles edge cases: few adapters, identical scores, missing data.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_toolkit.core.metrics import BenchmarkReport


def load_report(path: str) -> BenchmarkReport:
    return BenchmarkReport.from_json(path)


def generate_all_figures(report_path: str, output_dir: str):
    """Generate all figures for the paper."""
    report = load_report(report_path)
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })

    generate_score_comparison(report, output_dir)
    generate_token_comparison(report, output_dir)
    generate_overhead_comparison(report, output_dir)
    generate_heatmap(report, output_dir)
    generate_efficiency_scatter(report, output_dir)

    print(f"[OK] Generated 5 figures in {output_dir}")


def _adapter_order(adapters):
    """Consistent adapter ordering: flash first, then pro; single before multi."""
    preferred = [
        "v4-flash-single", "v4-flash-multi",
        "v4-pro-single", "v4-pro-multi",
        "single-agent", "single-agent-deepseek",
        "multi-step",
        "crewai", "autogen", "langgraph",
    ]
    ordered = []
    for p in preferred:
        if p in adapters:
            ordered.append(p)
    for a in adapters:
        if a not in ordered:
            ordered.append(a)
    return ordered


def _short_name(name: str) -> str:
    """Shorten adapter name for display."""
    replacements = {
        "v4-flash-single": "Flash\nSingle",
        "v4-flash-multi": "Flash\nMulti",
        "v4-pro-single": "Pro\nSingle",
        "v4-pro-multi": "Pro\nMulti",
        "single-agent": "Single\nAgent",
        "single-agent-deepseek": "Single\nAgent",
        "multi-step": "Multi\nStep",
        "crewai": "CrewAI",
        "autogen": "AutoGen",
        "langgraph": "LangGraph",
    }
    return replacements.get(name, name[:15])


def generate_score_comparison(report, output_dir):
    """Figure: Average scores by adapter with error bars."""
    adapters = _adapter_order(report.adapters)

    scores = {}
    for a in adapters:
        adapter_results = [r for r in report.results if r.adapter_name == a]
        if not adapter_results:
            scores[a] = {"mean": 0, "std": 0}
        else:
            scores[a] = {
                "mean": np.mean([r.score for r in adapter_results]),
                "std": np.std([r.score for r in adapter_results]) if len(adapter_results) > 1 else 0,
            }

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    names = [_short_name(a) for a in adapters]
    means = [scores[a]["mean"] for a in adapters]
    stds = [scores[a]["std"] for a in adapters]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"][:len(adapters)]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{mean:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_token_comparison(report, output_dir):
    """Figure: Token consumption comparison."""
    adapters = _adapter_order(report.adapters)

    data = {}
    for a in adapters:
        adapter_results = [r for r in report.results if r.adapter_name == a]
        data[a] = {
            "prompt": int(np.sum([r.metrics.prompt_tokens for r in adapter_results])),
            "completion": int(np.sum([r.metrics.completion_tokens for r in adapter_results])),
        }

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    names = [_short_name(a) for a in adapters]
    prompt_vals = [data[a]["prompt"] for a in adapters]
    completion_vals = [data[a]["completion"] for a in adapters]

    x = np.arange(len(names))
    ax.bar(x, prompt_vals, label="Prompt Tokens", color="#3498db", edgecolor="white", width=0.6)
    ax.bar(x, completion_vals, bottom=prompt_vals, label="Completion Tokens",
           color="#e74c3c", edgecolor="white", width=0.6)

    for i, (p, c) in enumerate(zip(prompt_vals, completion_vals)):
        total = p + c
        ax.text(i, total + max(prompt_vals)*0.03, f"{total/1000:.0f}K",
                ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Total Tokens")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "token_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_overhead_comparison(report, output_dir):
    """Figure: Time breakdown - LLM vs orchestration overhead."""
    adapters = _adapter_order(report.adapters)

    data = {}
    for a in adapters:
        adapter_results = [r for r in report.results if r.adapter_name == a]
        if not adapter_results:
            data[a] = {"llm": 0, "overhead": 0}
            continue
        avg_llm = np.mean([
            sum(c.duration_ms for c in r.metrics.call_log)
            for r in adapter_results
        ])
        avg_overhead = np.mean([r.metrics.orchestration_overhead_ms for r in adapter_results])
        data[a] = {"llm": avg_llm / 1000, "overhead": avg_overhead / 1000}

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    names = [_short_name(a) for a in adapters]
    llm_vals = [data[a]["llm"] for a in adapters]
    overhead_vals = [data[a]["overhead"] for a in adapters]

    x = np.arange(len(names))
    ax.bar(x, llm_vals, label="LLM Time", color="#2ecc71", edgecolor="white", width=0.6)
    ax.bar(x, overhead_vals, bottom=llm_vals, label="Overhead",
           color="#e67e22", edgecolor="white", width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "overhead_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_heatmap(report, output_dir):
    """Figure: Task x Adapter score heatmap."""
    adapters = _adapter_order(report.adapters)
    tasks = sorted(set(r.task_id for r in report.results))

    if len(tasks) == 0 or len(adapters) == 0:
        print("[WARNING] No data for heatmap, skipping")
        return

    # Build matrix
    matrix = np.zeros((len(tasks), len(adapters)))
    for i, task_id in enumerate(tasks):
        for j, adapter in enumerate(adapters):
            result = report.get_result(adapter, task_id)
            matrix[i, j] = result.score if result else 0.0

    # Handle edge case: all scores identical
    vmin, vmax = 0.0, 1.0
    unique_vals = np.unique(matrix)
    if len(unique_vals) == 1:
        vmin = max(0, unique_vals[0] - 0.1)
        vmax = min(1, unique_vals[0] + 0.1)
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(adapters)))
    ax.set_xticklabels([_short_name(a).replace("\n", " ") for a in adapters],
                       fontsize=7, rotation=20, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=7)

    # Annotations with smart color
    mid = (vmin + vmax) / 2
    for i in range(len(tasks)):
        for j in range(len(adapters)):
            color = "white" if matrix[i, j] < mid else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, label="Score", shrink=0.85)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_efficiency_scatter(report, output_dir):
    """Figure: Efficiency vs Interactions scatter plot."""
    adapters = _adapter_order(report.adapters)

    data = {}
    for a in adapters:
        adapter_results = [r for r in report.results if r.adapter_name == a]
        if not adapter_results:
            continue
        data[a] = {
            "efficiency": np.mean([r.metrics.efficiency_ratio for r in adapter_results]),
            "interactions": np.mean([r.metrics.agent_interactions for r in adapter_results]),
            "time": np.mean([r.metrics.total_time_ms for r in adapter_results]) / 1000,
            "score": np.mean([r.score for r in adapter_results]),
        }

    if not data:
        print("[WARNING] No data for efficiency scatter, skipping")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

    for i, (a, color) in enumerate(zip(adapters, colors)):
        if a not in data:
            continue
        d = data[a]
        size = max(80, d["time"] * 8)
        ax.scatter(d["interactions"], d["efficiency"],
                   s=size, color=color, alpha=0.7, edgecolors="black",
                   linewidth=0.5, zorder=5)
        ax.annotate(_short_name(a).replace("\n", " "),
                    (d["interactions"], d["efficiency"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)

    ax.set_xlabel("Avg Interactions per Task")
    ax.set_ylabel("Efficiency (Completion / Total Tokens)")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ensure reasonable axis limits — handle edge case where all x or y are identical
    all_x = [d["interactions"] for d in data.values()]
    all_y = [d["efficiency"] for d in data.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = (x_max - x_min) if (x_max - x_min) > 0 else 2
    y_range = (y_max - y_min) if (y_max - y_min) > 0 else 0.1
    ax.set_xlim(max(-0.5, x_min - x_range * 0.3), x_max + x_range * 0.3)
    ax.set_ylim(max(0, y_min - y_range * 0.3), min(1.05, y_max + y_range * 0.3))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "efficiency_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = glob.glob("experiments/results/*.json")
    if not results:
        print("[ERROR] No benchmark results found.")
        sys.exit(1)

    # Prefer final_combined, then combined_final, then multimodel
    preferred = [r for r in results if "final_combined" in r]
    if not preferred:
        preferred = [r for r in results if "combined_final" in r]
    if not preferred:
        preferred = [r for r in results if "multimodel" in r]
    if not preferred:
        preferred = results

    report_path = max(preferred, key=os.path.getmtime)
    print(f"Using: {report_path}")
    generate_all_figures(report_path, "paper/figures")

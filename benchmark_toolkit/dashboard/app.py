"""Streamlit dashboard for Multi-Agent Benchmark visualization.

Usage:
    streamlit run benchmark_toolkit/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark_toolkit.core.metrics import BenchmarkReport


st.set_page_config(
    page_title="Multi-Agent Benchmark",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_report(path: str) -> BenchmarkReport:
    return BenchmarkReport.from_json(path)


def main():
    st.title("🤖 Multi-Agent Collaboration Benchmark")
    st.caption("Compare multi-agent frameworks across standardized collaboration tasks")

    # Sidebar
    st.sidebar.header("📂 Data Source")
    uploaded = st.sidebar.file_uploader("Upload benchmark JSON report", type=["json"])

    # Also check for default paths
    default_paths = list(Path("experiments/results").glob("*.json"))
    if default_paths:
        selected = st.sidebar.selectbox(
            "Or select from results:",
            [p.name for p in default_paths],
        )
        if selected and not uploaded:
            report_path = str(Path("experiments/results") / selected)
            report = load_report(report_path)
        elif uploaded:
            content = uploaded.read().decode("utf-8")
            data = json.loads(content)
            report = BenchmarkReport(
                benchmark_id=data["benchmark_id"],
                timestamp=data["timestamp"],
                adapters=data["adapters"],
                tasks=data["tasks"],
                metadata=data.get("metadata", {}),
            )
        else:
            st.info("👈 Upload a benchmark report or run the CLI to generate one first.")
            st.code("mab run --output experiments/results/my_run.json", language="bash")
            return
    elif uploaded:
        content = uploaded.read().decode("utf-8")
        data = json.loads(content)
        report = BenchmarkReport(
            benchmark_id=data["benchmark_id"],
            timestamp=data["timestamp"],
            adapters=data["adapters"],
            tasks=data["tasks"],
            metadata=data.get("metadata", {}),
        )
        # Reconstruct results
        for r in data.get("results", []):
            from benchmark_toolkit.core.metrics import TaskResult, RunMetrics
            m = r["metrics"]
            metrics = RunMetrics(
                total_tokens=m["total_tokens"],
                prompt_tokens=m["prompt_tokens"],
                completion_tokens=m["completion_tokens"],
                total_time_ms=m["total_time_ms"],
                agent_interactions=m.get("agent_interactions", 0),
                tool_calls=m.get("tool_calls", 0),
                estimated_cost_usd=m.get("estimated_cost_usd", 0.0),
                orchestration_overhead_ms=m.get("orchestration_overhead_ms", 0.0),
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
                duration_ms=r.get("duration_ms", 0),
            ))
    else:
        st.info("👈 Upload a benchmark report or run the CLI to generate one first.")
        st.code("mab run --output experiments/results/my_run.json", language="bash")
        return

    st.sidebar.header("📊 Filters")
    selected_adapters = st.sidebar.multiselect(
        "Adapters",
        options=report.adapters,
        default=report.adapters,
    )
    selected_tasks = st.sidebar.multiselect(
        "Tasks",
        options=report.tasks,
        default=report.tasks,
    )

    st.sidebar.metric("Total Runs", len(report.results))
    st.sidebar.metric("Adapters", len(report.adapters))
    st.sidebar.metric("Tasks", len(report.tasks))

    # Filter results
    filtered = [
        r for r in report.results
        if r.adapter_name in selected_adapters and r.task_id in selected_tasks
    ]

    if not filtered:
        st.warning("No results match the selected filters.")
        return

    # Main content — tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Overview", "🔍 Per-Task", "💰 Cost Analysis", "📋 Raw Data"]
    )

    with tab1:
        _render_overview(report, filtered)

    with tab2:
        _render_per_task(report, filtered)

    with tab3:
        _render_cost_analysis(report, filtered)

    with tab4:
        st.dataframe(pd.DataFrame([r.__dict__ for r in filtered]))


def _render_overview(report: BenchmarkReport, filtered: list):
    """Render the overview tab."""
    col1, col2, col3 = st.columns(3)

    # Compute aggregate stats
    adapter_names = list(set(r.adapter_name for r in filtered))
    best_adapter = None
    best_score = -1
    for name in adapter_names:
        s = report.get_adapter_summary(name)
        if s and s.get("avg_score", 0) > best_score:
            best_score = s["avg_score"]
            best_adapter = name

    with col1:
        st.metric("🏆 Best Performer", best_adapter or "N/A",
                  f"Avg Score: {best_score:.2f}" if best_adapter else None)
    with col2:
        total_tokens = sum(r.metrics.total_tokens for r in filtered)
        st.metric("🎯 Total Tokens", f"{total_tokens:,}")
    with col3:
        total_cost = sum(r.metrics.estimated_cost_usd for r in filtered)
        st.metric("💵 Total Cost", f"${total_cost:.4f}")

    # Score comparison bar chart
    st.subheader("Average Score by Adapter")
    summaries = {}
    for name in adapter_names:
        s = report.get_adapter_summary(name)
        if s:
            summaries[name] = s

    if summaries:
        df_scores = pd.DataFrame({
            "Adapter": list(summaries.keys()),
            "Avg Score": [s["avg_score"] for s in summaries.values()],
            "Success Rate": [s["success_rate"] for s in summaries.values()],
        })
        fig = px.bar(
            df_scores, x="Adapter", y="Avg Score",
            color="Adapter",
            title="Average Task Score by Framework",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Efficiency comparison
    st.subheader("Collaboration Efficiency (Completion / Total Tokens)")
    df_eff = pd.DataFrame({
        "Adapter": list(summaries.keys()),
        "Efficiency": [s["avg_efficiency"] for s in summaries.values()],
        "Avg Interactions": [s["avg_interactions"] for s in summaries.values()],
    })
    fig2 = px.scatter(
        df_eff, x="Avg Interactions", y="Efficiency",
        size=[s["avg_time_ms"] / 1000 for s in summaries.values()],
        text="Adapter",
        title="Efficiency vs Interaction Count (bubble size = time)",
    )
    fig2.update_traces(textposition="top center")
    st.plotly_chart(fig2, use_container_width=True)


def _render_per_task(report: BenchmarkReport, filtered: list):
    """Render per-task comparison."""
    tasks = list(set(r.task_id for r in filtered))
    adapters = list(set(r.adapter_name for r in filtered))

    if not tasks or not adapters:
        return

    # Heatmap
    st.subheader("Score Heatmap: Tasks × Adapters")
    heatmap_data = []
    for task_id in tasks:
        row = {"Task": task_id[:40]}
        for adapter_name in adapters:
            result = report.get_result(adapter_name, task_id)
            row[adapter_name] = result.score if result else 0.0
        heatmap_data.append(row)

    df_heat = pd.DataFrame(heatmap_data).set_index("Task")

    fig = px.imshow(
        df_heat,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Task Scores by Adapter",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-task detail
    st.subheader("Task Details")
    selected_task = st.selectbox("Select task:", tasks)
    if selected_task:
        for adapter_name in adapters:
            result = report.get_result(adapter_name, selected_task)
            if result:
                with st.expander(f"{adapter_name} — Score: {result.score:.2f}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tokens", f"{result.metrics.total_tokens:,}")
                    c2.metric("Time", f"{result.metrics.total_time_ms/1000:.1f}s")
                    c3.metric("Cost", f"${result.metrics.estimated_cost_usd:.4f}")
                    if result.output:
                        st.text_area("Output", result.output[:2000], height=200)
                    if result.error:
                        st.error(result.error)


def _render_cost_analysis(report: BenchmarkReport, filtered: list):
    """Render cost and token analysis."""
    adapter_names = list(set(r.adapter_name for r in filtered))

    st.subheader("Token Consumption Comparison")
    col1, col2 = st.columns(2)

    with col1:
        # Total tokens per adapter
        token_data = []
        for name in adapter_names:
            adapter_results = [r for r in filtered if r.adapter_name == name]
            total_tokens = sum(r.metrics.total_tokens for r in adapter_results)
            total_prompt = sum(r.metrics.prompt_tokens for r in adapter_results)
            total_completion = sum(r.metrics.completion_tokens for r in adapter_results)
            token_data.append({
                "Adapter": name,
                "Prompt Tokens": total_prompt,
                "Completion Tokens": total_completion,
            })

        df_tokens = pd.DataFrame(token_data)
        fig = px.bar(
            df_tokens, x="Adapter", y=["Prompt Tokens", "Completion Tokens"],
            title="Token Breakdown by Adapter",
            barmode="stack",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cost comparison
        cost_data = []
        for name in adapter_names:
            adapter_results = [r for r in filtered if r.adapter_name == name]
            total_cost = sum(r.metrics.estimated_cost_usd for r in adapter_results)
            avg_cost = total_cost / len(adapter_results) if adapter_results else 0
            cost_data.append({"Adapter": name, "Total Cost (USD)": total_cost,
                              "Avg Cost per Task (USD)": avg_cost})

        df_cost = pd.DataFrame(cost_data)
        fig = px.bar(
            df_cost, x="Adapter", y="Total Cost (USD)",
            title="Total API Cost by Adapter",
            color="Adapter",
            text_auto=".4f",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time breakdown
    st.subheader("Time Analysis")
    time_data = []
    for name in adapter_names:
        adapter_results = [r for r in filtered if r.adapter_name == name]
        avg_time = sum(r.metrics.total_time_ms for r in adapter_results) / len(adapter_results)
        avg_llm_time = sum(
            sum(c.duration_ms for c in r.metrics.call_log)
            for r in adapter_results
        ) / len(adapter_results)
        avg_overhead = sum(r.metrics.orchestration_overhead_ms for r in adapter_results) / len(adapter_results)
        time_data.append({
            "Adapter": name,
            "Avg LLM Time (s)": avg_llm_time / 1000,
            "Avg Overhead (s)": avg_overhead / 1000,
        })

    df_time = pd.DataFrame(time_data)
    fig = px.bar(
        df_time, x="Adapter", y=["Avg LLM Time (s)", "Avg Overhead (s)"],
        title="Time Breakdown: LLM vs Orchestration Overhead",
        barmode="stack",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

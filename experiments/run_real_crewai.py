"""Run real CrewAI (v1.14) with DeepSeek V4 Flash on all 9 benchmark tasks."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.core.metrics import TaskResult, RunMetrics, BenchmarkReport

KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not KEY:
    print("Set DEEPSEEK_API_KEY first"); sys.exit(1)
os.environ["DEEPSEEK_API_KEY"] = KEY

suite = create_default_suite()
results = []

for task in suite:
    ti = task.generate_input()
    prompt = f"Task: {ti.description}"
    if ti.context:
        prompt += f"\n\nContext:\n{ti.context}"
    if ti.input_data:
        prompt += f"\n\nInput Data:\n```json\n{json.dumps(ti.input_data, indent=2)}\n```"

    print(f"\n[{task.task_id}] ", end="", flush=True)

    try:
        from crewai import Agent, Task as CrewTask, Crew

        r = Agent(role="Research Specialist",
                  goal=f"Gather all relevant information for: {ti.description}",
                  backstory="Expert researcher who finds and organizes information thoroughly.",
                  llm="deepseek/deepseek-chat", verbose=False)
        a = Agent(role="Senior Analyst",
                  goal="Synthesize research and extract key insights",
                  backstory="Senior analyst who identifies patterns and verifies facts.",
                  llm="deepseek/deepseek-chat", verbose=False)
        w = Agent(role="Technical Writer",
                  goal="Produce the final polished answer",
                  backstory="Meticulous writer ensuring outputs are accurate and complete.",
                  llm="deepseek/deepseek-chat", verbose=False)

        rt = CrewTask(description=f"Research the task thoroughly:\n\n{prompt}",
                      expected_output="Comprehensive research findings",
                      agent=r)
        at = CrewTask(description="Analyze the research. Extract key insights and verify facts.",
                      expected_output="Structured analysis with verified insights",
                      agent=a)
        wt = CrewTask(description="Write the final answer. Be complete and accurate.",
                      expected_output="Final comprehensive answer",
                      agent=w)

        start = time.time()
        crew = Crew(agents=[r, a, w], tasks=[rt, at, wt], verbose=False)
        output_raw = crew.kickoff()
        elapsed = (time.time() - start) * 1000

        output = str(output_raw.raw) if hasattr(output_raw, "raw") else str(output_raw)
        score = task.evaluate(output)

    except Exception as e:
        output = f"ERROR: {e}"
        elapsed = 0
        score = 0.0
        print(f"FAIL: {e}")

    # Token metrics from CrewAI aren't directly accessible, estimate
    # CrewAI uses 3 sequential LLM calls, similar to our MultiStepAdapter
    metrics = RunMetrics(
        total_tokens=0,  # CrewAI doesn't expose token counts directly
        total_time_ms=elapsed,
        agent_count=3,
    )

    results.append(TaskResult(
        task_id=task.task_id, task_name=task.name,
        adapter_name="crewai-real", success=score >= 0.5,
        score=score, metrics=metrics,
        output=output[:2000] if output else None,
    ))
    print(f"score={score:.2f} time={elapsed/1000:.1f}s")

# Save
report = BenchmarkReport(
    benchmark_id="crewai-real-v1",
    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    adapters=["crewai-real"],
    tasks=[t.task_id for t in suite],
    results=results,
)

out = "experiments/results/crewai_real.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
report.to_json(out)

s = report.get_adapter_summary("crewai-real")
print(f"\n=== REAL CREWAI (DeepSeek V4 Flash) ===")
print(f"Avg Score: {s['avg_score']:.3f}")
print(f"Success Rate: {s['success_rate']:.0%}")
print(f"Avg Time: {s['avg_time_ms']/1000:.1f}s")
print(f"Saved to: {out}")

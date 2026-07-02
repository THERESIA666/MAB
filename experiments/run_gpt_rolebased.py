"""Run role-based adapter with GPT-4o-mini and GPT-4o via proxy."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.adapters.role_based_adapter import RoleBasedAdapter
from benchmark_toolkit.core.adapter import AdapterConfig
from benchmark_toolkit.core.runner import BenchmarkRunner

KEY = os.environ.get("OPENAI_PROXY_KEY", "")
BASE = os.environ.get("OPENAI_PROXY_BASE", "")

suite = create_default_suite()
all_results = []

for model, name in [("gpt-4o-mini", "gpt4o-mini-role"), ("gpt-4o", "gpt4o-role")]:
    config = AdapterConfig(adapter_name=name, llm_model=model,
                           llm_provider="openai", api_key=KEY,
                           api_base=BASE, temperature=0.0, max_tokens=4096)
    adapter = RoleBasedAdapter(config)
    print(f"\n{'='*50}")
    print(f"Running: {name} ({model})")
    print(f"{'='*50}")
    runner = BenchmarkRunner(adapters=[adapter], repeat=1, timeout_seconds=300, verbose=True)
    report = runner.run(suite)
    all_results.extend(report.results)

# Merge with existing
from benchmark_toolkit.core.metrics import BenchmarkReport
existing = BenchmarkReport.from_json("experiments/results/final_combined.json")
merged = BenchmarkReport(
    benchmark_id="final-all-11", timestamp=existing.timestamp,
    adapters=list(existing.adapters) + ["gpt4o-mini-role", "gpt4o-role"],
    tasks=sorted(set(r.task_id for r in list(existing.results) + all_results)),
    results=list(existing.results) + all_results,
    metadata={"total_runs": len(existing.results) + len(all_results)},
)

out = "experiments/results/final_combined.json"
merged.to_json(out)

print("\n=== ALL 11 CONFIGURATIONS ===")
for a in merged.adapters:
    s = merged.get_adapter_summary(a)
    if s:
        print(f"{a:<25} score={s['avg_score']:.3f} succ={s['success_rate']:.0%} "
              f"tokens={s['total_tokens']:,} cost=${s['total_cost_usd']:.4f}")

# Cross-provider role-based comparison
print("\n=== ROLE-BASED: Flash vs GPT ===")
for name in ["role-based-flash", "gpt4o-mini-role", "gpt4o-role"]:
    s = merged.get_adapter_summary(name)
    if s:
        print(f"  {name}: score={s['avg_score']:.3f} succ={s['success_rate']:.0%}")
print(f"\n[OK] {len(all_results)} new runs, saved to {out}")

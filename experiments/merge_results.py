"""Merge DeepSeek + OpenAI results into a combined report for the paper."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_toolkit.core.metrics import BenchmarkReport

# Load both reports
ds = BenchmarkReport.from_json("experiments/results/multimodel_comparison.json")
oa = BenchmarkReport.from_json("experiments/results/openai_proxy.json")

# Merge
all_results = list(ds.results) + list(oa.results)
all_adapters = list(ds.adapters) + list(oa.adapters)
all_tasks = sorted(set(r.task_id for r in all_results))

merged = BenchmarkReport(
    benchmark_id="cross-provider-combined",
    timestamp=ds.timestamp,
    adapters=all_adapters,
    tasks=all_tasks,
    results=all_results,
    metadata={
        "providers": ["deepseek", "openai-proxy"],
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o-mini", "gpt-4o"],
        "total_runs": len(all_results),
    },
)

out = "experiments/results/combined_final.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
merged.to_json(out)

print("=== COMBINED CROSS-PROVIDER RESULTS ===")
print(f"Total runs: {len(all_results)}")
print(f"Adapters: {all_adapters}")
print()

# Summary table
print(f"{'Configuration':<25} {'Score':>6} {'Succ':>6} {'Tokens':>8} {'Time(s)':>8} {'Cost':>8}")
print("-" * 68)
for name in all_adapters:
    s = merged.get_adapter_summary(name)
    if s:
        print(f"{name:<25} {s['avg_score']:>6.3f} {s['success_rate']:>5.0%} "
              f"{s['total_tokens']:>8,} {s['avg_time_ms']/1000:>7.1f}s "
              f"${s['total_cost_usd']:>7.4f}")

# Best single vs best multi per provider
print("\n=== COLLABORATION BONUS BY PROVIDER ===")
pairs = [
    ("DeepSeek Flash", "v4-flash-single", "v4-flash-multi"),
    ("DeepSeek Pro", "v4-pro-single", "v4-pro-multi"),
    ("GPT-4o Mini", "gpt4o-mini-single", "gpt4o-mini-multi"),
    ("GPT-4o", "gpt4o-single", "gpt4o-multi"),
]
print(f"{'Model':<16} {'Single':>7} {'Multi':>7} {'Delta':>7} {'Token×':>7} {'Worth?':>8}")
print("-" * 58)
for label, single, multi in pairs:
    ss = merged.get_adapter_summary(single)
    ms = merged.get_adapter_summary(multi)
    if ss and ms:
        delta = ms["avg_score"] - ss["avg_score"]
        token_ratio = ms["total_tokens"] / ss["total_tokens"] if ss["total_tokens"] else 0
        worth = "YES" if delta > 0.08 else "MAYBE" if delta > 0.03 else "NO"
        print(f"{label:<16} {ss['avg_score']:>7.3f} {ms['avg_score']:>7.3f} "
              f"{'+' if delta>0 else ''}{delta:>6.3f} {token_ratio:>6.1f}x {worth:>8}")

print(f"\n[OK] Saved to: {out}")

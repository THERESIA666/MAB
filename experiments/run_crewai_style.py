"""Run role-based (CrewAI-style) adapter with DeepSeek V4 Flash.

Adds the role-based collaboration pattern to complement our existing
single-agent and multi-step results.
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.adapters.role_based_adapter import RoleBasedAdapter
from benchmark_toolkit.core.adapter import AdapterConfig
from benchmark_toolkit.core.runner import BenchmarkRunner

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def main():
    suite = create_default_suite()

    config = AdapterConfig(
        adapter_name="role-based-flash",
        llm_model="deepseek-v4-flash",
        llm_provider="deepseek",
        api_key=API_KEY,
        temperature=0.0,
        max_tokens=4096,
    )
    adapter = RoleBasedAdapter(config)

    print("=" * 50)
    print("CrewAI-Style Role-Based Adapter (DeepSeek V4 Flash)")
    print("=" * 50)
    print(f"Tasks: {len(suite)}")

    runner = BenchmarkRunner(adapters=[adapter], repeat=1,
                             timeout_seconds=300, verbose=True)
    report = runner.run(suite)

    out = "experiments/results/role_based_flash.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report.to_json(out)

    s = report.get_adapter_summary("role-based-flash")
    print(f"\nRole-Based Flash: score={s['avg_score']:.3f} succ={s['success_rate']:.0%} "
          f"tokens={s['total_tokens']:,} time={s['avg_time_ms']/1000:.1f}s "
          f"cost=${s['total_cost_usd']:.4f}")

    # Quick comparison with existing results
    print("\n--- Quick Comparison ---")
    ds_path = "experiments/results/combined_final.json"
    if os.path.exists(ds_path):
        from benchmark_toolkit.core.metrics import BenchmarkReport
        combined = BenchmarkReport.from_json(ds_path)
        for name in ["v4-flash-single", "v4-flash-multi", "role-based-flash"]:
            if name == "role-based-flash":
                summ = s
            else:
                summ = combined.get_adapter_summary(name)
            if summ:
                print(f"  {name:<25} score={summ['avg_score']:.3f} "
                      f"succ={summ['success_rate']:.0%} "
                      f"tokens={summ['total_tokens']:,} "
                      f"cost=${summ['total_cost_usd']:.4f}")

    print(f"\n[OK] Saved to: {out}")


if __name__ == "__main__":
    main()

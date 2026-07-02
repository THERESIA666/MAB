"""Run OpenAI models via proxy — single vs multi-step comparison.

Adds gpt-4o-mini and gpt-4o data to complement existing DeepSeek results.
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_toolkit.tasks import create_default_suite
from benchmark_toolkit.adapters.single_agent import SingleAgentAdapter
from benchmark_toolkit.adapters.multi_step_adapter import MultiStepAdapter
from benchmark_toolkit.core.adapter import AdapterConfig
from benchmark_toolkit.core.runner import BenchmarkRunner
from benchmark_toolkit.core.metrics import BenchmarkReport

PROXY_KEY = os.environ.get("OPENAI_PROXY_KEY", "")
PROXY_BASE = "https://api.ohmygpt.com/v1"


def make_config(name, model):
    return AdapterConfig(
        adapter_name=name,
        llm_model=model,
        llm_provider="openai",
        api_key=PROXY_KEY,
        api_base=PROXY_BASE,
        temperature=0.0,
        max_tokens=4096,
    )


def main():
    suite = create_default_suite()

    # OpenAI models via proxy
    configs = [
        ("gpt4o-mini-single", "gpt-4o-mini", SingleAgentAdapter),
        ("gpt4o-mini-multi", "gpt-4o-mini", MultiStepAdapter),
        ("gpt4o-single", "gpt-4o", SingleAgentAdapter),
        ("gpt4o-multi", "gpt-4o", MultiStepAdapter),
    ]

    print("=" * 60)
    print("OpenAI Proxy Benchmark: gpt-4o-mini vs gpt-4o")
    print("=" * 60)
    print(f"Configs: {len(configs)}, Tasks: {len(suite)}")
    print(f"Total runs: {len(configs) * len(suite)}")
    print()

    all_results = []

    for name, model, adapter_cls in configs:
        cfg = make_config(name, model)
        adapter = adapter_cls(cfg)
        print(f"\n{'='*50}")
        print(f"Running: {name} ({model})")
        print(f"{'='*50}")

        runner = BenchmarkRunner(adapters=[adapter], repeat=1,
                                 timeout_seconds=300, verbose=True)
        report = runner.run(suite)
        all_results.extend(report.results)

    # Merge results
    adapter_names = [c[0] for c in configs]
    final = BenchmarkReport(
        benchmark_id="openai-proxy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        adapters=adapter_names,
        tasks=sorted(set(r.task_id for r in all_results)),
        results=all_results,
        metadata={"provider": "openai-proxy", "models": ["gpt-4o-mini", "gpt-4o"]},
    )

    out = "experiments/results/openai_proxy.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    final.to_json(out)

    # Summary
    print("\n" + "=" * 60)
    print("OPENAI RESULTS")
    print("=" * 60)
    for name in adapter_names:
        s = final.get_adapter_summary(name)
        if s:
            print(f"  {name:<22} score={s['avg_score']:.3f} succ={s['success_rate']:.0%} "
                  f"tokens={s['total_tokens']:,} time={s['avg_time_ms']/1000:.1f}s "
                  f"cost=${s['total_cost_usd']:.4f}")

    # Comparison with DeepSeek (if available)
    ds_path = "experiments/results/multimodel_comparison.json"
    if os.path.exists(ds_path):
        ds = BenchmarkReport.from_json(ds_path)
        print("\n--- Cross-Provider Comparison ---")
        print(f"{'Model':<25} {'Score':>6} {'Succ':>6} {'Tokens':>8} {'Time':>8} {'Cost':>8}")
        print("-" * 65)
        for name in adapter_names:
            s = final.get_adapter_summary(name)
            if s:
                print(f"{name:<25} {s['avg_score']:>6.3f} {s['success_rate']:>5.0%} "
                      f"{s['total_tokens']:>8,} {s['avg_time_ms']/1000:>7.1f}s "
                      f"${s['total_cost_usd']:>7.4f}")
        for name in ds.adapters:
            s = ds.get_adapter_summary(name)
            if s:
                print(f"{name:<25} {s['avg_score']:>6.3f} {s['success_rate']:>5.0%} "
                      f"{s['total_tokens']:>8,} {s['avg_time_ms']/1000:>7.1f}s "
                      f"${s['total_cost_usd']:>7.4f}")

    print(f"\n[OK] Results saved to: {out}")


if __name__ == "__main__":
    main()

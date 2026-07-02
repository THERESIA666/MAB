"""Unit tests for core modules — metrics, task, adapter config."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from benchmark_toolkit.core.metrics import RunMetrics, TaskResult, BenchmarkReport
from benchmark_toolkit.core.task import Task, TaskInput, TaskSuite
from benchmark_toolkit.core.adapter import AdapterConfig, InstrumentedLLM


class TestRunMetrics:
    def test_default_values(self):
        m = RunMetrics()
        assert m.total_tokens == 0
        assert m.prompt_tokens == 0
        assert m.completion_tokens == 0
        assert m.total_time_ms == 0.0
        assert m.estimated_cost_usd == 0.0
        assert m.efficiency_ratio == 0.0

    def test_efficiency_ratio(self):
        m = RunMetrics(total_tokens=100, completion_tokens=40)
        assert m.efficiency_ratio == 0.4

    def test_efficiency_zero_division(self):
        m = RunMetrics(total_tokens=0, completion_tokens=0)
        assert m.efficiency_ratio == 0.0

    def test_tokens_per_interaction(self):
        m = RunMetrics(total_tokens=500, agent_interactions=5)
        assert m.tokens_per_interaction == 100.0

    def test_tokens_per_interaction_zero(self):
        m = RunMetrics(total_tokens=500, agent_interactions=0)
        assert m.tokens_per_interaction == 500.0

    def test_to_dict(self):
        m = RunMetrics(
            total_tokens=1000, prompt_tokens=600, completion_tokens=400,
            total_time_ms=5000.0, agent_interactions=3,
            estimated_cost_usd=0.005, orchestration_overhead_ms=1000.0,
        )
        d = m.to_dict()
        assert d["total_tokens"] == 1000
        assert d["prompt_tokens"] == 600
        assert d["completion_tokens"] == 400
        assert d["agent_interactions"] == 3
        assert d["estimated_cost_usd"] == 0.005


class TestTaskResult:
    def test_creation(self):
        metrics = RunMetrics(total_tokens=500, total_time_ms=3000)
        tr = TaskResult(
            task_id="test-1", task_name="Test Task",
            adapter_name="single-agent", success=True, score=0.85,
            metrics=metrics, output="test output",
        )
        assert tr.task_id == "test-1"
        assert tr.success is True
        assert tr.score == 0.85


class TestBenchmarkReport:
    def _make_report(self):
        report = BenchmarkReport(
            benchmark_id="test-001",
            timestamp="2026-06-25T00:00:00",
            adapters=["adapter-a", "adapter-b"],
            tasks=["task-1", "task-2"],
        )
        # Add results
        for adapter in ["adapter-a", "adapter-b"]:
            for task_id in ["task-1", "task-2"]:
                score = 0.8 if adapter == "adapter-a" else 0.6
                report.results.append(TaskResult(
                    task_id=task_id, task_name=f"Task {task_id}",
                    adapter_name=adapter, success=score >= 0.5,
                    score=score,
                    metrics=RunMetrics(
                        total_tokens=1000, prompt_tokens=600,
                        completion_tokens=400, total_time_ms=5000,
                        estimated_cost_usd=0.005,
                    ),
                ))
        return report

    def test_get_result(self):
        report = self._make_report()
        r = report.get_result("adapter-a", "task-1")
        assert r is not None
        assert r.score == 0.8

    def test_get_result_missing(self):
        report = self._make_report()
        r = report.get_result("nonexistent", "task-1")
        assert r is None

    def test_get_adapter_summary(self):
        report = self._make_report()
        summary = report.get_adapter_summary("adapter-a")
        assert summary["adapter"] == "adapter-a"
        assert summary["tasks_completed"] == 2
        assert summary["avg_score"] == 0.8
        assert summary["success_rate"] == 1.0

    def test_get_ranking(self):
        report = self._make_report()
        ranking = report.get_ranking("avg_score")
        assert ranking[0][0] == "adapter-a"
        assert ranking[1][0] == "adapter-b"
        assert ranking[0][1] > ranking[1][1]

    def test_json_roundtrip(self):
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            report.to_json(f.name)
            path = f.name
        try:
            loaded = BenchmarkReport.from_json(path)
            assert loaded.benchmark_id == "test-001"
            assert len(loaded.results) == 4
            assert loaded.get_result("adapter-a", "task-1").score == 0.8
        finally:
            os.unlink(path)


class TestTask:
    def _make_simple_task(self):
        def input_gen():
            return TaskInput(task_id="simple", description="Say hello")
        def evaluator(output: str) -> float:
            return 1.0 if "hello" in output.lower() else 0.0
        return Task(
            task_id="simple", name="Simple Task",
            description="Say hello",
            category="qa", difficulty="easy",
            input_generator=input_gen, evaluator=evaluator,
            min_agents_recommended=1,
        )

    def test_generate_input(self):
        task = self._make_simple_task()
        inp = task.generate_input()
        assert inp.task_id == "simple"
        assert "hello" in inp.description.lower()

    def test_evaluate_success(self):
        task = self._make_simple_task()
        assert task.evaluate("Hello world") == 1.0

    def test_evaluate_failure(self):
        task = self._make_simple_task()
        assert task.evaluate("Goodbye") == 0.0

    def test_evaluate_exception(self):
        def broken_evaluator(output):
            raise RuntimeError("eval failed")
        task = Task(
            task_id="broken", name="Broken",
            description="desc", category="qa", difficulty="easy",
            input_generator=lambda: TaskInput(task_id="b", description="d"),
            evaluator=broken_evaluator,
        )
        assert task.evaluate("anything") == 0.0


class TestTaskSuite:
    def _make_suite(self):
        def make_task(tid, cat, diff):
            return Task(
                task_id=tid, name=tid, description=tid,
                category=cat, difficulty=diff,
                input_generator=lambda: TaskInput(task_id=tid, description=tid),
                evaluator=lambda o: 1.0,
            )
        return TaskSuite([
            make_task("t1", "qa", "easy"),
            make_task("t2", "qa", "hard"),
            make_task("t3", "code", "medium"),
            make_task("t4", "data", "hard"),
        ])

    def test_len_and_iter(self):
        suite = self._make_suite()
        assert len(suite) == 4
        assert len(list(suite)) == 4

    def test_get(self):
        suite = self._make_suite()
        assert suite.get("t1").task_id == "t1"
        assert suite.get("nonexistent") is None

    def test_filter_category(self):
        suite = self._make_suite()
        qa = suite.filter(category="qa")
        assert len(qa) == 2

    def test_filter_difficulty(self):
        suite = self._make_suite()
        hard = suite.filter(difficulty="hard")
        assert len(hard) == 2

    def test_filter_combined(self):
        suite = self._make_suite()
        result = suite.filter(category="qa", difficulty="easy")
        assert len(result) == 1
        assert result[0].task_id == "t1"

    def test_categories(self):
        suite = self._make_suite()
        assert set(suite.categories) == {"qa", "code", "data"}

    def test_task_ids(self):
        suite = self._make_suite()
        assert suite.task_ids == ["t1", "t2", "t3", "t4"]


class TestAdapterConfig:
    def test_defaults(self):
        config = AdapterConfig(adapter_name="test")
        assert config.adapter_name == "test"
        assert config.llm_model == "deepseek-v4-flash"
        assert config.llm_provider == "deepseek"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.thinking_mode is False

    def test_resolve_api_key_direct(self):
        config = AdapterConfig(adapter_name="test", api_key="sk-test")
        assert config.resolve_api_key() == "sk-test"

    def test_resolve_api_key_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        config = AdapterConfig(adapter_name="test", llm_provider="deepseek")
        assert config.resolve_api_key() == "sk-from-env"

    def test_resolve_api_base_deepseek(self):
        config = AdapterConfig(adapter_name="test", llm_provider="deepseek")
        assert config.resolve_api_base() == "https://api.deepseek.com"

    def test_resolve_api_base_custom(self):
        config = AdapterConfig(adapter_name="test", llm_provider="deepseek",
                               api_base="https://custom.api.com")
        assert config.resolve_api_base() == "https://custom.api.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

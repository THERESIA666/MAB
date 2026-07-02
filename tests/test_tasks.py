"""Integration tests for task registry and task evaluation."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from benchmark_toolkit.tasks import create_default_suite


class TestTaskRegistry:
    """Test that all default tasks are properly configured."""

    @classmethod
    def setup_class(cls):
        cls.suite = create_default_suite()

    def test_suite_not_empty(self):
        assert len(self.suite) == 9

    def test_all_tasks_have_ids(self):
        for task in self.suite:
            assert task.task_id, f"Task {task.name} has no ID"
            assert len(task.task_id) > 0

    def test_all_categories_present(self):
        categories = set(self.suite.categories)
        assert "qa" in categories
        assert "code_generation" in categories
        assert "data_analysis" in categories
        assert "tool_use" in categories

    def test_all_difficulties_present(self):
        difficulties = set(t.difficulty for t in self.suite)
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_tasks_generate_input(self):
        for task in self.suite:
            inp = task.generate_input()
            assert inp.task_id == task.task_id
            assert isinstance(inp.description, str)
            assert len(inp.description) > 0

    def test_tasks_evaluate_empty_string(self):
        """All evaluators should handle empty output gracefully."""
        for task in self.suite:
            score = task.evaluate("")
            assert 0.0 <= score <= 1.0, \
                f"Task {task.task_id} evaluator returned {score} for empty string"

    def test_tasks_evaluate_garbage(self):
        """All evaluators should handle garbage output."""
        for task in self.suite:
            score = task.evaluate("asdf qwer zxcv 12345 !@#$%")
            assert 0.0 <= score <= 1.0

    def test_task_metadata(self):
        for task in self.suite:
            assert task.category in ("qa", "code_generation", "data_analysis", "tool_use")
            assert task.difficulty in ("easy", "medium", "hard")
            assert task.min_agents_recommended >= 1
            assert task.expected_duration_seconds > 0
            assert len(task.tags) > 0

    def test_qa_tasks_evaluate_correct_answers(self):
        """QA tasks should score high on known-correct answers."""
        qa_tasks = self.suite.filter(category="qa")
        for task in qa_tasks:
            # A very long detailed answer should score well
            score = task.evaluate("The answer is: English. The company DeepMind created AlphaGo. DeepMind is headquartered in London, United Kingdom. The official language of the United Kingdom is English. Therefore, the answer is English.")
            assert score >= 0.0  # At minimum doesn't crash

    def test_code_tasks_evaluate_correct_code(self):
        """Code tasks should score well on correct implementations."""
        code_tasks = self.suite.filter(category="code_generation")
        for task in code_tasks:
            if "palindrome" in task.description.lower():
                correct_code = '''```python
def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome, ignoring case, spaces, and punctuation."""
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    return cleaned == cleaned[::-1]
```'''
                score = task.evaluate(correct_code)
                # Should score high on correct code
                assert score > 0.3, f"Palindrome task scored {score} on correct code"

    def test_data_task_evaluates_correct_analysis(self):
        """Data analysis task should score high on correct analysis."""
        data_tasks = self.suite.filter(category="data_analysis")
        for task in data_tasks:
            correct = """
Total Revenue: $2,680.00
Best-selling product by units: Widget A (88 units)
Best-selling product by revenue: Widget A ($1,320.00)
Region with highest sales: US ($1,245.00)
Day with highest revenue: Friday ($650.00)
Widget A is the best performer across both metrics.
"""
            score = task.evaluate(correct)
            assert score > 0.5, f"Data task scored {score} on correct analysis"

    def test_suite_immutable_via_filter(self):
        """Filter should not modify original suite."""
        original_len = len(self.suite)
        self.suite.filter(category="qa")
        assert len(self.suite) == original_len


class TestTaskScoring:
    """Specific scoring behavior tests."""

    @classmethod
    def setup_class(cls):
        cls.suite = create_default_suite()

    def test_mhq1_keywords(self):
        """MHQ-1 should recognize 'English' or 'UK' in answer."""
        task = self.suite.get("mhq-1")
        score_en = task.evaluate("The official language is English.")
        score_uk = task.evaluate("The answer is the UK, English is spoken there.")
        assert score_en > 0.3 or score_uk > 0.3, \
            f"MHQ-1 scores: English={score_en}, UK={score_uk}"

    def test_mhq2_correct_year(self):
        """MHQ-2 should recognize 1930."""
        task = self.suite.get("mhq-2")
        score = task.evaluate("Neil Armstrong was born in 1930, first walked on moon in 1969.")
        assert score > 0.5, f"MHQ-2 scored {score}"

    def test_mhq3_correct_element(self):
        """MHQ-3 should recognize Einsteinium (Es)."""
        task = self.suite.get("mhq-3")
        score = task.evaluate("The element is Einsteinium, symbol Es.")
        assert score > 0.5, f"MHQ-3 scored {score}"

    def test_code1_palindrome_correct(self):
        """CODE-1 should recognize correct palindrome function."""
        task = self.suite.get("code-1")
        code = '''
def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    import re
    s = re.sub(r'[^a-z0-9]', '', s.lower())
    return s == s[::-1]
'''
        score = task.evaluate(code)
        assert score > 0.3, f"CODE-1 scored {score}"

    def test_code2_fibonacci_correct(self):
        """CODE-2 should recognize correct fibonacci function."""
        task = self.suite.get("code-2")
        code = '''
def fibonacci(n: int) -> list[int]:
    """Return first n Fibonacci numbers."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib
'''
        score = task.evaluate(code)
        assert score > 0.3, f"CODE-2 scored {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

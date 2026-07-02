"""Task definitions for the multi-agent benchmark suite."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TaskInput:
    """Input for a benchmark task."""
    task_id: str
    description: str
    input_data: dict[str, Any] = field(default_factory=dict)
    context: str = ""  # additional context or instructions


@dataclass
class TaskOutput:
    """Expected structure of task output (for validation)."""
    task_id: str
    output: Any
    score: float  # 0.0 - 1.0
    evaluation_notes: str = ""


@dataclass
class Task:
    """A benchmark task definition.

    Each task represents a collaboration-intensive problem that requires
    multiple agents working together to solve effectively.
    """
    task_id: str
    name: str
    description: str
    category: str  # "qa", "code_generation", "data_analysis", "tool_use"
    difficulty: str  # "easy", "medium", "hard"

    # Core task components
    input_generator: Callable[[], TaskInput]
    evaluator: Callable[[str], float]  # scores output string -> 0.0-1.0

    # Task metadata
    min_agents_recommended: int = 2
    expected_duration_seconds: int = 60
    tags: list[str] = field(default_factory=list)
    ground_truth: Optional[str] = None

    def generate_input(self) -> TaskInput:
        return self.input_generator()

    def evaluate(self, output: str) -> float:
        try:
            return self.evaluator(output)
        except Exception:
            return 0.0


class TaskSuite:
    """A collection of benchmark tasks."""

    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self._by_id = {t.task_id: t for t in tasks}

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def get(self, task_id: str) -> Optional[Task]:
        return self._by_id.get(task_id)

    def filter(self, category: str = None, difficulty: str = None) -> list[Task]:
        result = self.tasks
        if category:
            result = [t for t in result if t.category == category]
        if difficulty:
            result = [t for t in result if t.difficulty == difficulty]
        return result

    @property
    def categories(self) -> list[str]:
        return list(set(t.category for t in self.tasks))

    @property
    def task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks]

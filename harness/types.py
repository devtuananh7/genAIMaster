from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Task:
    task_id: int
    text: str
    test_list: list[str]
    test_imports: list[str]


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    stdout: str
    stderr: str
    traceback: str
    failed_test: str | None
    passed_count: int
    total_count: int
    duration_ms: int


class Strategy(Protocol):
    name: str

    def solve(self, task: Task) -> str:
        """Return Python source code for the given task."""

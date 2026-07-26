"""
reuse_rag/tasks.py
==================
RepoTask — bài toán "dùng lại API": như harness.Task nhưng kèm API đích
(target_module / target_name) và tên hàm wrapper mà model phải viết.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from harness.types import Task


@dataclass(frozen=True)
class RepoTask:
    task_id: int
    wrapper: str
    text: str
    target_module: str
    target_name: str
    test_list: list[str]
    test_imports: list[str]

    @property
    def target_fqn(self) -> str:
        return f"{self.target_module}.{self.target_name}"

    def to_harness_task(self) -> Task:
        """Chuyển sang harness.Task để tái dùng executor M0."""
        return Task(
            task_id=self.task_id,
            text=self.text,
            test_list=list(self.test_list),
            test_imports=list(self.test_imports),
        )


def load_repo_tasks(path: str | Path) -> list[RepoTask]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [
        RepoTask(
            task_id=int(item["task_id"]),
            wrapper=str(item["wrapper"]),
            text=str(item["text"]),
            target_module=str(item["target_module"]),
            target_name=str(item["target_name"]),
            test_list=[str(t) for t in item["test_list"]],
            test_imports=[str(t) for t in item.get("test_imports", [])],
        )
        for item in payload
    ]

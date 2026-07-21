from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from harness.types import ExecutionResult, Task

TIMEOUT_SECONDS = 10
MAX_TRACEBACK_LINES = 15


def _shorten_traceback(stderr: str) -> str:
    lines = stderr.strip().splitlines()
    if len(lines) <= MAX_TRACEBACK_LINES:
        return "\n".join(lines)
    return "\n".join(lines[-MAX_TRACEBACK_LINES:])


def _script_for(code: str, task: Task, assertion: str) -> str:
    parts: list[str] = []
    parts.extend(import_line for import_line in task.test_imports if import_line.strip())
    parts.append(code.strip())
    parts.append(assertion.strip())
    return "\n\n".join(parts) + "\n"


def run(code: str, task: Task) -> ExecutionResult:
    start = time.monotonic()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    passed_count = 0
    total_count = len(task.test_list)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "candidate_test.py"

        for assertion in task.test_list:
            tmp_path.write_text(_script_for(code, task, assertion), encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    timeout=TIMEOUT_SECONDS,
                    capture_output=True,
                    text=True,
                    cwd=tmpdir,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                return ExecutionResult(
                    status="timeout",
                    stdout=stdout,
                    stderr=stderr,
                    traceback=_shorten_traceback(stderr),
                    failed_test=assertion,
                    passed_count=passed_count,
                    total_count=total_count,
                    duration_ms=duration_ms,
                )

            stdout_parts.append(completed.stdout)
            stderr_parts.append(completed.stderr)

            if completed.returncode == 0:
                passed_count += 1
                continue

            stderr = completed.stderr
            if "SyntaxError" in stderr:
                status = "error_syntax"
                failed_test = None
            elif "AssertionError" in stderr:
                status = "fail_assert"
                failed_test = assertion
            else:
                status = "error_runtime"
                failed_test = assertion

            duration_ms = int((time.monotonic() - start) * 1000)
            return ExecutionResult(
                status=status,
                stdout="".join(stdout_parts),
                stderr="".join(stderr_parts),
                traceback=_shorten_traceback(stderr),
                failed_test=failed_test,
                passed_count=passed_count,
                total_count=total_count,
                duration_ms=duration_ms,
            )

    duration_ms = int((time.monotonic() - start) * 1000)
    return ExecutionResult(
        status="pass",
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        traceback="",
        failed_test=None,
        passed_count=passed_count,
        total_count=total_count,
        duration_ms=duration_ms,
    )

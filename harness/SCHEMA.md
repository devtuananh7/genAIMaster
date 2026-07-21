# ExecutionResult Schema

`ExecutionResult` is the frozen contract between the harness and later modules.

Fields:

- `status`: one of `pass`, `fail_assert`, `error_runtime`, `error_syntax`, `timeout`.
- `stdout`: captured standard output from the subprocess.
- `stderr`: captured standard error from the subprocess.
- `traceback`: stderr traceback shortened to at most 15 lines.
- `failed_test`: the first failing assert string, or `null`.
- `passed_count`: number of asserts that passed.
- `total_count`: total number of asserts evaluated.
- `duration_ms`: wall-clock execution time in milliseconds.

The executor runs each assert separately so `failed_test` can identify the first failing test.

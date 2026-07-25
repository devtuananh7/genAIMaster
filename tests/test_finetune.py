"""Tests cho Stage A của M5 (deterministic — không GPU, không API thật)."""

from __future__ import annotations

from finetune.build_dataset import format_example
from finetune.collect_seeds import collect
from finetune.filter_runnable import is_runnable
from finetune.gen_teacher import _parse_teacher_json, generate_for_seed, make_mock_chat


# --- teacher JSON parsing --------------------------------------------------
def test_parse_teacher_json_valid():
    raw = 'Sure!\n{"problem": "p", "solution": "def f(): pass", "tests": ["assert True"]}\n'
    obj = _parse_teacher_json(raw)
    assert obj and obj["problem"] == "p" and obj["tests"] == ["assert True"]


def test_parse_teacher_json_missing_keys():
    assert _parse_teacher_json('{"problem": "p"}') is None
    assert _parse_teacher_json("not json") is None


def test_generate_for_seed_with_mock():
    seed = {"seed_id": "s1", "source_repo": "boltons", "snippet": "def g(x): return x"}
    sample = generate_for_seed(
        seed, temperature=0.7, max_tokens=256, chat_fn=make_mock_chat(), model=None
    )
    assert sample["seed_id"] == "s1"
    assert "double_all" in sample["solution"]
    assert sample["tests"]


# --- filter via M0 executor ------------------------------------------------
def test_filter_keeps_runnable():
    good = {
        "problem": "double",
        "solution": "def double_all(nums):\n    return [n*2 for n in nums]",
        "tests": ["assert double_all([1,2]) == [2,4]"],
    }
    ok, status = is_runnable(good)
    assert ok and status == "pass"


def test_filter_drops_wrong_solution():
    bad = {
        "problem": "double",
        "solution": "def double_all(nums):\n    return nums",  # sai
        "tests": ["assert double_all([1,2]) == [2,4]"],
    }
    ok, status = is_runnable(bad)
    assert not ok and status != "pass"


# --- dataset formatting ----------------------------------------------------
def test_format_example_structure():
    sample = {
        "problem": "double a list",
        "solution": "def double_all(nums):\n    return [n*2 for n in nums]",
        "tests": ["assert double_all([1]) == [2]"],
    }
    text = format_example(sample, eos="<EOS>")
    assert "### System:" in text and "### User:" in text and "### Assistant:" in text
    assert "```python" in text and text.rstrip().endswith("<EOS>")


# --- seed collection -------------------------------------------------------
def test_collect_seeds_from_boltons():
    seeds = collect(["boltons"])
    assert len(seeds) >= 30
    s = seeds[0]
    assert s["source_repo"] == "boltons" and s["license"].startswith("BSD")
    assert 1 <= len(s["snippet"].splitlines()) <= 15

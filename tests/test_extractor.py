from harness.extractor import extract_code


def test_extracts_python_fence():
    text = "Answer:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(text) == "def add(a, b):\n    return a + b"


def test_extracts_plain_fence():
    text = "```\ndef square(x):\n    return x * x\n```"
    assert extract_code(text) == "def square(x):\n    return x * x"


def test_falls_back_to_def_line():
    text = "Here is the function:\ndef cube(x):\n    return x ** 3"
    assert extract_code(text) == "def cube(x):\n    return x ** 3"


def test_prefers_first_python_block_over_later_blocks():
    text = (
        "```python\ndef first():\n    return 1\n```\n"
        "```python\ndef second():\n    return 2\n```"
    )
    assert extract_code(text) == "def first():\n    return 1"


def test_prefers_python_block_over_plain_block():
    text = (
        "```\nnot python\n```\n"
        "```python\ndef target():\n    return True\n```"
    )
    assert extract_code(text) == "def target():\n    return True"


def test_strips_surrounding_prose_and_code_whitespace():
    text = "Use this:\n```python\n\n  def value():\n      return 7\n\n```\nDone."
    assert extract_code(text) == "def value():\n      return 7"

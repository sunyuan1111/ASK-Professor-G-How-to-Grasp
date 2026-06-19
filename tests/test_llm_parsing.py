from ask_professor_g.llm.gemini import GeminiClient
from ask_professor_g.llm.openai_compatible import OpenAICompatibleClient
from ask_professor_g.llm.parser import extract_json, extract_python


def test_extract_json_from_plain_and_fenced_text():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("```json\n{\"a\": 2}\n```") == {"a": 2}
    assert extract_json("prefix {\"a\": 3} suffix") == {"a": 3}


def test_extract_python_from_fenced_text():
    code = extract_python("```python\nimport numpy as np\n\ndef calculate_loss(a, b):\n    return 0\n```")
    assert code.startswith("import numpy")
    assert "calculate_loss" in code


def test_provider_construction_does_not_require_hardcoded_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gemini = GeminiClient.from_settings({"model": "x"})
    openai = OpenAICompatibleClient.from_settings({"model": "y", "base_url": "http://localhost"})
    assert gemini.api_key is None
    assert openai.api_key is None


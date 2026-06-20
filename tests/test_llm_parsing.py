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
    monkeypatch.delenv("API_KEY", raising=False)
    gemini = GeminiClient.from_settings({"model": "x"})
    openai = OpenAICompatibleClient.from_settings({"model": "y", "base_url": "http://localhost"})
    assert gemini.api_key is None
    assert openai.api_key is None


def test_openai_compatible_accepts_short_api_env_aliases(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("API_BASE_URL", "http://localhost/v1")
    client = OpenAICompatibleClient.from_settings({"model": "y"})
    assert client.api_key == "test-key"
    assert client.base_url == "http://localhost/v1"

def test_extract_python_keeps_constants_before_function_after_preface():
    raw = (
        "Short note before code.import numpy as np\n\n"
        "_STAGE1_TARGETS = np.array([[1.0, 2.0, 3.0]])\n\n"
        "def calculate_loss(pose_mat: np.ndarray, point_cloud: np.ndarray) -> float:\n"
        "    return float(_STAGE1_TARGETS[0, 0])\n"
    )
    code = extract_python(raw)
    assert code.startswith("import numpy as np")
    assert "_STAGE1_TARGETS" in code.split("def calculate_loss", 1)[0]

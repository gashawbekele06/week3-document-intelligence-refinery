import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.provider_keys import resolve_provider_keys


def test_resolve_provider_keys_maps_openrouter_key_from_anthropic_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-or-v1-demo-key")

    openrouter_api_key, openai_api_key, anthropic_api_key = resolve_provider_keys()

    assert openrouter_api_key == "sk-or-v1-demo-key"
    assert openai_api_key == ""
    assert anthropic_api_key == ""


def test_resolve_provider_keys_keeps_real_anthropic_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-real-anthropic-key")

    openrouter_api_key, openai_api_key, anthropic_api_key = resolve_provider_keys()

    assert openrouter_api_key == ""
    assert openai_api_key == ""
    assert anthropic_api_key == "sk-ant-api03-real-anthropic-key"
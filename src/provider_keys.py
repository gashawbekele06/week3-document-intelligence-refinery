from __future__ import annotations

import os
from typing import Tuple


def resolve_provider_keys() -> Tuple[str, str, str]:
    """Normalize provider credentials across entrypoints."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if anthropic_api_key.startswith("sk-or-"):
        if not openrouter_api_key:
            openrouter_api_key = anthropic_api_key
        anthropic_api_key = ""

    return openrouter_api_key, openai_api_key, anthropic_api_key


def apply_provider_key_aliases() -> None:
    """Mirror normalized keys back into the active process environment."""
    openrouter_api_key, _, anthropic_api_key = resolve_provider_keys()

    if openrouter_api_key and not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    if not anthropic_api_key and os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-or-"):
        os.environ.pop("ANTHROPIC_API_KEY", None)
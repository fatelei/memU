from __future__ import annotations

from typing import Any, Mapping

from dify_plugin.interfaces.tool import ToolProvider


class MemUToolProvider(ToolProvider):
    """Validates credentials for MemU tools.

    Expected credentials (from provider.yaml):
    - api_key: LLM API key (required)
    - base_url: LLM base URL (optional)
    - chat_model: chat model name (optional)
    - embed_model: embedding model name (optional)
    """

    def _validate_credentials(self, credentials: Mapping[str, Any]):  # type: ignore[override]
        api_key = credentials.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key is required")
        # basic sanity checks for optional fields
        base_url = credentials.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("base_url must be a string if provided")
        for key in ("chat_model", "embed_model"):
            val = credentials.get(key)
            if val is not None and not isinstance(val, str):
                raise ValueError(f"{key} must be a string if provided")

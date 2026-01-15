from __future__ import annotations

import asyncio
from typing import Any, Generator

from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.interfaces.tool import Tool

from memu.app.service import MemoryService
from memu.app.settings import (
    BlobConfig,
    DatabaseConfig,
    LLMConfig,
    LLMProfilesConfig,
    MemorizeConfig,
    RetrieveConfig,
    UserConfig,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _build_service(credentials: dict[str, Any]) -> MemoryService:
    base_url = credentials.get("base_url") or "https://api.openai.com/v1"
    api_key = credentials.get("api_key")
    chat_model = credentials.get("chat_model") or "gpt-4o-mini"
    embed_model = credentials.get("embed_model") or "text-embedding-3-small"

    llm_profiles = LLMProfilesConfig.model_validate(
        {
            "default": LLMConfig(base_url=base_url, api_key=api_key, chat_model=chat_model, client_backend="sdk"),
            "embedding": LLMConfig(base_url=base_url, api_key=api_key, embed_model=embed_model, client_backend="sdk"),
        }
    )

    service = MemoryService(
        llm_profiles=llm_profiles,
        blob_config=BlobConfig(),
        database_config=DatabaseConfig(),
        memorize_config=MemorizeConfig(),
        retrieve_config=RetrieveConfig(),
        user_config=UserConfig(),
    )
    return service


class MemorizeTool(Tool):
    """MemU memorize tool: ingest a resource and extract memories."""

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:  # type: ignore[override]
        resource_url = tool_parameters.get("resource_url")
        modality = tool_parameters.get("modality")
        user = tool_parameters.get("user")

        if not isinstance(resource_url, str) or not resource_url.strip():
            raise ValueError("resource_url is required")
        if not isinstance(modality, str) or modality not in {"conversation", "document", "audio", "image", "video"}:
            raise ValueError("invalid modality")
        if user is not None and not isinstance(user, dict):
            raise ValueError("user must be an object if provided")

        svc = _build_service(self.runtime.credentials)

        # emit a start log
        start_log = self.create_log_message(
            label="memu.memorize.start",
            data={"resource_url": resource_url, "modality": modality},
        )
        yield start_log

        result = _run(svc.memorize(resource_url=resource_url, modality=modality, user=user))

        yield self.create_json_message(result)
        yield self.finish_log_message(start_log, data={"ok": True})

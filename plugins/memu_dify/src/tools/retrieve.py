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


def _build_service(credentials: dict[str, Any], method: str | None) -> MemoryService:
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

    retrieve_cfg = RetrieveConfig()
    if method in {"rag", "llm"}:
        retrieve_cfg.method = method  # type: ignore[assignment]

    service = MemoryService(
        llm_profiles=llm_profiles,
        blob_config=BlobConfig(),
        database_config=DatabaseConfig(),
        memorize_config=MemorizeConfig(),
        retrieve_config=retrieve_cfg,
        user_config=UserConfig(),
    )
    return service


class RetrieveTool(Tool):
    """MemU retrieve tool: search memories by query."""

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:  # type: ignore[override]
        query = tool_parameters.get("query")
        method = tool_parameters.get("method")
        where = tool_parameters.get("where")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        if method is not None and method not in {"rag", "llm"}:
            raise ValueError("method must be 'rag' or 'llm'")
        if where is not None and not isinstance(where, dict):
            raise ValueError("where must be an object if provided")

        svc = _build_service(self.runtime.credentials, method)

        queries = [
            {
                "role": "user",
                "content": {"text": query},
            }
        ]

        start_log = self.create_log_message(
            label="memu.retrieve.start",
            data={"method": method or "rag", "query": query},
        )
        yield start_log

        result = _run(svc.retrieve(queries=queries, where=where))

        yield self.create_json_message(result)
        yield self.finish_log_message(start_log, data={"ok": True})

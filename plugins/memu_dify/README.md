# MemU Dify Plugin

This plugin adds two Dify-compatible tools for using MemU inside Dify agents or workflows:

- memorize_resource: ingest a resource (conversation/document/audio/image/video) and extract structured memories
- retrieve_memories: search MemU for categories/items/resources relevant to a query

## How to use in Dify

1. Package the plugin folder `plugins/memu_dify` and load it with Dify Plugin Runner.
2. Configure provider credentials:
   - api_key: LLM API key (e.g., OpenAI)
   - base_url: optional LLM base URL
   - chat_model: default `gpt-4o-mini`
   - embed_model: default `text-embedding-3-small`
3. In an Agent or Workflow, add the provider "MemU" and enable tools:
   - memorize_resource
   - retrieve_memories

### Example (Agent)
- The agent can call `memorize_resource` when it sees a file/URL to store.
- For answering questions, the agent calls `retrieve_memories` with the user query to fetch context (categories/items/resources) and use them in its reply.

### Example (Workflow)
- Node A: a file URL is produced -> call `memorize_resource` with modality.
- Node B: user query arrives -> call `retrieve_memories(method=rag)` -> pass results to LLM node.

Outputs are JSON containing items/resources/categories, aligned with MemU's `MemoryService` results.

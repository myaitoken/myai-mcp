"""
MyAi MCP Server

Exposes the MyAi decentralized inference network as MCP tools so any
MCP-compatible agent (Claude, Cursor, etc.) can run inference, browse
available models and providers, and query network/wallet state.

Environment variables:
  MYAI_API_KEY   — your MyAi API key (required for inference + wallet tools)
  MYAI_BASE_URL  — coordinator base URL (default: https://api.myaitoken.io)
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

BASE_URL = os.environ.get("MYAI_BASE_URL", "https://api.myaitoken.io").rstrip("/")
API_KEY  = os.environ.get("MYAI_API_KEY", "")

app = Server("myai")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers(require_auth: bool = True) -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    elif require_auth:
        raise ValueError("MYAI_API_KEY is not set. Export it before starting the server.")
    return h


async def _get(path: str, auth: bool = True) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}{path}", headers=_headers(auth))
        r.raise_for_status()
        return r.json()


async def _post(path: str, body: dict, auth: bool = True) -> Any:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{BASE_URL}{path}", headers=_headers(auth), json=body)
        r.raise_for_status()
        return r.json()


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=f"Error: {msg}")]


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="myai_chat",
        description=(
            "Run a chat completion through the MyAi decentralized inference network. "
            "Jobs are routed to the best available provider node based on model, hardware "
            "capability, and on-chain reputation. Providers earn MYAI tokens per completion. "
            "Use this instead of a centralised inference API when you want decentralised, "
            "censorship-resistant inference or when you want to contribute to the network."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": (
                        "Model name to run (e.g. 'qwen2.5-14b-awq', 'llama3.1:8b'). "
                        "Call myai_list_models first to see what's available on the network right now."
                    ),
                },
                "messages": {
                    "type": "array",
                    "description": "OpenAI-format message array: [{role, content}, ...]",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (0–2). Default 0.7.",
                    "default": 0.7,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate. Default 1024.",
                    "default": 1024,
                },
            },
            "required": ["model", "messages"],
        },
    ),
    Tool(
        name="myai_list_models",
        description=(
            "List all AI models currently available on the MyAi network, with the "
            "provider agents that serve each model. Use this before myai_chat to pick "
            "a model that is actually online."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="myai_list_agents",
        description=(
            "List active provider nodes on the MyAi network. Returns each agent's ID, "
            "hardware specs (GPU/CPU, VRAM), models it serves, on-chain wallet address, "
            "reputation score, and current status. Useful for understanding network "
            "capacity or selecting a specific provider."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "online_only": {
                    "type": "boolean",
                    "description": "If true, return only agents that are currently online. Default true.",
                    "default": True,
                }
            },
        },
    ),
    Tool(
        name="myai_network_stats",
        description=(
            "Get live network statistics: number of active agents, jobs completed in the "
            "last 24h, total MYAI distributed, models available, and routing activity. "
            "Good for a health check or to understand current network capacity before "
            "submitting a batch of jobs."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="myai_wallet_balance",
        description=(
            "Check the MYAI token balance and recent transactions for a wallet address "
            "on Base mainnet. If no address is provided, returns the balance for the "
            "wallet associated with your API key."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "EVM wallet address (0x...). Optional — defaults to your key's wallet.",
                }
            },
        },
    ),
    Tool(
        name="myai_hire_agent",
        description=(
            "Delegate an inference job directly to a specific named provider agent on the "
            "MyAi network. Use myai_list_agents first to find agent IDs. This bypasses "
            "the MARL router and pins the job to one agent — useful when you need a "
            "specific model or hardware configuration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID from myai_list_agents (e.g. 'hal9000', 'tanner')",
                },
                "model": {"type": "string", "description": "Model to run on the agent."},
                "messages": {
                    "type": "array",
                    "description": "OpenAI-format message array.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                },
            },
            "required": ["agent_id", "model", "messages"],
        },
    ),
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "myai_chat":
            body = {
                "model": arguments["model"],
                "messages": arguments["messages"],
                "temperature": arguments.get("temperature", 0.7),
                "max_tokens": arguments.get("max_tokens", 1024),
                "stream": False,
            }
            result = await _post("/v1/chat/completions", body)
            # Extract just the assistant content for readability
            choices = result.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                return [TextContent(type="text", text=content + f"\n\n[tokens: {usage}]")]
            return _ok(result)

        elif name == "myai_list_models":
            result = await _get("/v1/models", auth=bool(API_KEY))
            models = result.get("data", result)
            summary = [
                {"id": m.get("id"), "owned_by": m.get("owned_by"), "created": m.get("created")}
                for m in (models if isinstance(models, list) else [])
            ]
            return _ok({"models": summary, "count": len(summary)})

        elif name == "myai_list_agents":
            online_only = arguments.get("online_only", True)
            if online_only:
                result = await _get("/api/v1/agents/gpu/online")
            else:
                result = await _get("/api/v1/agents/jobs/all")
            return _ok(result)

        elif name == "myai_network_stats":
            activity = await _get("/api/v1/agents/activity", auth=bool(API_KEY))
            models   = await _get("/v1/models", auth=bool(API_KEY))
            model_count = len(models.get("data", []))
            return _ok({
                "activity": activity,
                "model_count": model_count,
                "base_url": BASE_URL,
            })

        elif name == "myai_wallet_balance":
            address = arguments.get("address")
            path = f"/api/v1/wallet/balance"
            if address:
                path += f"?address={address}"
            result = await _get(path)
            return _ok(result)

        elif name == "myai_hire_agent":
            body = {
                "model": arguments["model"],
                "messages": arguments["messages"],
                "stream": False,
            }
            agent_id = arguments["agent_id"]
            result = await _post(f"/api/v1/agents/{agent_id}/dispatch", body)
            return _ok(result)

        else:
            return _err(f"Unknown tool: {name}")

    except httpx.HTTPStatusError as e:
        return _err(f"HTTP {e.response.status_code}: {e.response.text[:400]}")
    except Exception as e:
        return _err(str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def run() -> None:
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()

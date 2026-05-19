"""
MyAi MCP Server

Exposes the MyAi decentralized inference network as MCP tools so any
MCP-compatible agent (Claude, Cursor, etc.) can run inference, browse
available models and providers, and query network/wallet state.

Zero-config — if MYAI_API_KEY isn't set, a free-tier key is auto-issued on
first call and cached at ~/.myai/key. 100 free completions, no signup.

Environment variables:
  MYAI_API_KEY   — your MyAi API key. Optional; auto-issued if unset.
  MYAI_BASE_URL  — coordinator base URL (default: https://api.myaitoken.io)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from pathlib import Path
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

# Cache the auto-issued key here so we only mint once per machine.
_KEY_CACHE_PATH = Path(os.environ.get("MYAI_KEY_CACHE", "")) if os.environ.get("MYAI_KEY_CACHE") \
                  else Path.home() / ".myai" / "key"

app = Server("myai")


# ── Zero-friction free-key bootstrap ─────────────────────────────────────────

def _machine_fingerprint() -> str:
    """Stable per-machine ID. Same machine = same fingerprint = same key.

    Built from: MAC address (uuid.getnode), hostname, platform, py version.
    Hashed so the raw values never leave the box.
    """
    parts = [
        hex(uuid.getnode()),
        socket.gethostname(),
        platform.system(),
        platform.machine(),
        sys.version.split()[0],
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _load_cached_key() -> str:
    try:
        if _KEY_CACHE_PATH.exists():
            v = _KEY_CACHE_PATH.read_text().strip()
            if v.startswith("myai"):
                return v
    except Exception:
        pass
    return ""


def _save_cached_key(key: str) -> None:
    try:
        _KEY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _KEY_CACHE_PATH.write_text(key)
        try:
            _KEY_CACHE_PATH.chmod(0o600)
        except Exception:
            pass
    except Exception:
        pass


async def _bootstrap_key() -> str:
    """Issue (or recall) a free-tier key. Idempotent — called at most once
    per server lifetime; result is cached in API_KEY module global + on disk."""
    global API_KEY
    if API_KEY:
        return API_KEY
    cached = _load_cached_key()
    if cached:
        API_KEY = cached
        return API_KEY
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{BASE_URL}/v1/keys/auto-issue",
            json={
                "machine_id": _machine_fingerprint(),
                "client": "myai-mcp/0.2.0",
            },
        )
        r.raise_for_status()
        key = r.json()["api_key"]
    API_KEY = key
    _save_cached_key(key)
    return key


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _headers(require_auth: bool = True) -> dict:
    h = {"Content-Type": "application/json"}
    if require_auth and not API_KEY:
        # First call without a key — auto-issue a free-tier one.
        await _bootstrap_key()
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


async def _get(path: str, auth: bool = True) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}{path}", headers=await _headers(auth))
        r.raise_for_status()
        return r.json()


async def _post(path: str, body: dict, auth: bool = True) -> Any:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{BASE_URL}{path}", headers=await _headers(auth), json=body)
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
            # These endpoints are public — no API key required.
            online_only = arguments.get("online_only", True)
            if online_only:
                result = await _get("/api/v1/agents/gpu/online", auth=False)
            else:
                result = await _get("/api/v1/agents/jobs/all", auth=False)
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

# myai-mcp

<!-- mcp-name: io.github.myaitoken/myai-mcp -->

MCP server for the [MyAi](https://myaitoken.io) decentralized AI inference network.

Gives any MCP-compatible agent (Claude Desktop, Cursor, etc.) access to:

- **`myai_chat`** — run inference through the decentralized network
- **`myai_list_models`** — see what models are online right now
- **`myai_list_agents`** — browse active provider nodes and their hardware
- **`myai_network_stats`** — live network health and job activity
- **`myai_wallet_balance`** — check MYAI token balance for any address
- **`myai_hire_agent`** — pin a job to a specific provider node

## Install

```bash
uvx myai-mcp
```

Or with pip:

```bash
pip install myai-mcp
```

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "myai": {
      "command": "uvx",
      "args": ["myai-mcp"],
      "env": {
        "MYAI_API_KEY": "your-api-key-here",
        "MYAI_BASE_URL": "https://api.myaitoken.io"
      }
    }
  }
}
```

Get an API key at [myaitoken.io](https://myaitoken.io).

## Example usage (once connected)

Ask Claude:

> "Use myai_list_models to see what's available, then use myai_chat with qwen2.5-14b to summarize this document."

> "Check the MyAi network stats and tell me how many providers are online."

> "What's the MYAI balance for 0x454F85B685BBe6C1257b218eE40782d17dFacA9b?"

## Links

- [Provider guide](https://myaitoken.io/earn) — run a node and earn MYAI
- [Agent registry](https://myaitoken.io/agents) — browse the live network
- [Docs](https://myaitoken.io/docs)
- [Discord](https://discord.gg/2mUwktpRS)

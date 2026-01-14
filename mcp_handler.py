import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}

    async def connect_to_server(self):
        """Connect to all MCP servers defined in config."""
        with open('mcp_config.json') as f:
            config = json.load(f)

        mcp_servers = config.get('mcpServers', {})
        sse_servers = config.get('sseServers', {})

        for name, server_config in mcp_servers.items():
            await self._connect_stdio(name, server_config)

        for name, server_config in sse_servers.items():
            await self._connect_sse(name, server_config)

        if self.sessions:
            self.session = list(self.sessions.values())[0]

    async def _connect_stdio(self, name: str, config: dict):
        """Connect to an MCP server via stdio."""
        server_params = StdioServerParameters(
            command=config['command'],
            args=config.get('args', []),
            env=config.get('env')
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.sessions[name] = session

        tools = await session.list_tools()
        print(f"\nConnected to {name} (stdio) with tools:", [t.name for t in tools.tools])

    async def _connect_sse(self, name: str, config: dict):
        """Connect to an MCP server via SSE."""
        url = config['url']

        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url)
        )
        read, write = sse_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.sessions[name] = session

        tools = await session.list_tools()
        print(f"\nConnected to {name} (sse) with tools:", [t.name for t in tools.tools])

    async def list_all_tools(self):
        """List tools from all connected servers."""
        all_tools = []
        for name, session in self.sessions.items():
            response = await session.list_tools()
            for tool in response.tools:
                all_tools.append((name, tool))
        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool, finding the right session automatically."""
        for name, session in self.sessions.items():
            response = await session.list_tools()
            tool_names = [t.name for t in response.tools]
            if tool_name in tool_names:
                return await session.call_tool(tool_name, arguments)
        raise ValueError(f"Tool {tool_name} not found in any connected server")

    async def cleanup(self):
        """Clean up all connections."""
        await self.exit_stack.aclose()


if __name__ == "__main__":
    async def main():
        client = MCPClient()
        await client.connect_to_server()
        tools = await client.list_all_tools()
        print("All tools:", [(name, t.name) for name, t in tools])

    asyncio.run(main())

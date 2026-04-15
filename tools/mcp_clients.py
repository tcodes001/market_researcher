# tools/mcp_client.py

"""
MCP Client — persistent connection to Tavily MCP Server.

Responsibilities:
- Start Tavily MCP server as a subprocess via npx
- Maintain a persistent stdio connection and session
- Expose LangChain-compatible tools to the agent layer
- Clean up all processes on shutdown
"""
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class MCPClient:
    def __init__(self):
        npx_cmd = "npx.cmd" if sys.platform == "windows" else "npx"
        self.server_params = StdioServerParameters(
            command=npx_cmd,
            args=["-y","tavily-mcp@latest"],
            env={
                **os.environ,
                "TAVILY_API_KEY" : os.getenv("TAVILY_API_KEY")
            }
        )
        self.tools = []
        self._exit_stack = AsyncExitStack()
    
    async def initialize(self) -> list:
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.tools = await load_mcp_tools(session)
        return self.tools
    
    async def cleanup(self):
        await self._exit_stack.aclose()

    def get_tools(self) -> list:
        return self.tools
    
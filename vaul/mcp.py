"""Utilities for working with MCP servers."""

from typing import Any, Dict, List

import asyncio

from mcp import ClientSession

from .decorators import tool_call, ToolCall


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _create_mcp_tool(session: ClientSession, tool: Any) -> ToolCall:
    """Create a ``ToolCall`` wrapper around an MCP tool."""
    name = getattr(tool, "name")
    description = getattr(tool, "description", "") or ""
    schema = (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
        or getattr(tool, "parameters", None)
        or {}
    )

    async def _async_call(**kwargs):
        return await session.call_tool(name=name, arguments=kwargs)

    def mcp_function(**kwargs):
        return _run_async(_async_call(**kwargs))

    mcp_function.__name__ = name
    mcp_function.__doc__ = description

    tool_call_wrapper = tool_call(mcp_function)
    tool_call_wrapper.tool_call_schema = {
        "name": name,
        "description": description,
        "parameters": schema,
    }
    return tool_call_wrapper


def tools_from_mcp(session: ClientSession) -> List[ToolCall]:
    """Load tools from an MCP ``ClientSession``."""
    response = _run_async(session.list_tools())
    tools_data = getattr(response, "tools", response)

    tools: List[ToolCall] = []
    for item in tools_data:
        tools.append(_create_mcp_tool(session, item))
    return tools

"""Utilities for working with MCP servers."""

from typing import Any, List, Dict, Optional, Callable
import asyncio
import concurrent.futures
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from .decorators import tool_call, ToolCall

logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running - create and use a new one
        return asyncio.run(coro)
    
    # Event loop already running - handle nested case
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        # Fall back to thread-based execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


def _extract_tool_metadata(tool: Any) -> tuple[str, str, dict]:
    """Extract name, description, and schema from various tool formats."""
    # Get name (required)
    name = getattr(tool, "name", None) or tool.get("name", "")
    if not name:
        raise ValueError("Tool must have a name")
    
    # Get description (optional)
    description = getattr(tool, "description", "") or tool.get("description", "")
    
    # Get schema (optional) - try common attribute names
    schema = {}
    for attr in ["inputSchema", "input_schema", "parameters"]:
        schema = getattr(tool, attr, None) or tool.get(attr, {})
        if schema:
            break
    
    return name, description, schema


def _extract_result_content(result: Any) -> Any:
    """Extract content from various MCP result formats."""
    # Check for 'content' attribute
    if hasattr(result, 'content'):
        content = result.content
        
        # Handle list of content items
        if isinstance(content, list) and content:
            item = content[0]
            # Try to extract text or data from the item
            return getattr(item, 'text', None) or getattr(item, 'data', None) or str(item)
        
        return str(content)
    
    # Check for 'result' attribute
    if hasattr(result, 'result'):
        return result.result
    
    # Return as-is if no special handling needed
    return result


def _parse_tools_response(response: Any) -> List[Any]:
    """Parse tools from various response formats."""
    # Try different ways to extract tools list
    if hasattr(response, 'tools'):
        return response.tools
    if isinstance(response, dict) and 'tools' in response:
        return response['tools']
    if isinstance(response, list):
        return response
    
    return []


def _create_tool_wrapper(
    name: str,
    description: str,
    schema: dict,
    async_call_func: Callable
) -> ToolCall:
    """Create a ToolCall wrapper with the given metadata and async call function."""
    # Create synchronous wrapper
    def sync_wrapper(**kwargs):
        return _run_async(async_call_func(**kwargs))
    
    # Set function metadata
    sync_wrapper.__name__ = name
    sync_wrapper.__doc__ = description
    
    # Create and configure the tool
    tool = tool_call(sync_wrapper)
    tool.tool_call_schema = {
        "name": name,
        "description": description,
        "parameters": schema,
    }
    
    return tool


def _create_tool_from_metadata(
    tool_metadata: Any,
    create_async_call: Callable[[str], Callable]
) -> ToolCall:
    """Create a tool from metadata and an async call factory."""
    name, description, schema = _extract_tool_metadata(tool_metadata)
    async_call = create_async_call(name)
    return _create_tool_wrapper(name, description, schema, async_call)

async def _load_tools_async(
    session: ClientSession,
    create_tool: Callable[[Any], ToolCall]
) -> List[ToolCall]:
    """Load tools from an MCP session asynchronously."""
    response = await session.list_tools()
    tools_data = _parse_tools_response(response)
    
    tools = []
    for item in tools_data:
        try:
            tool = create_tool(item)
            tools.append(tool)
        except Exception as e:
            logger.warning(f"Failed to create tool: {e}")
    
    return tools

def tools_from_mcp(session: ClientSession) -> List[ToolCall]:
    """
    Load tools from an existing MCP ClientSession.
    
    Args:
        session: An active MCP ClientSession
        
    Returns:
        List of ToolCall objects that can be added to a Toolkit
    """
    def create_tool(tool_metadata: Any) -> ToolCall:
        """Create a tool that uses the existing session."""
        def create_async_call(name: str):
            async def call(**kwargs):
                result = await session.call_tool(name=name, arguments=kwargs)
                return _extract_result_content(result)
            return call
        
        return _create_tool_from_metadata(tool_metadata, create_async_call)
    
    return _run_async(_load_tools_async(session, create_tool))


def tools_from_mcp_url(
    url: str,
    headers: Dict[str, str] = {}
) -> List[ToolCall]:
    """
    Load tools from an MCP server URL (SSE endpoint).
    
    Args:
        url: The SSE endpoint URL
        headers: HTTP headers
        
    Returns:
        List of ToolCall objects that can be added to a Toolkit
    """
    async def load():
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                def create_tool(tool_metadata: Any) -> ToolCall:
                    """Create a tool that reconnects for each call over SSE."""
                    def create_async_call(name: str):
                        async def call(**kwargs):
                            async with sse_client(url, headers=headers) as (r, w):
                                async with ClientSession(r, w) as session:
                                    await session.initialize()
                                    result = await session.call_tool(name=name, arguments=kwargs)
                                    return _extract_result_content(result)
                        return call
                    
                    return _create_tool_from_metadata(tool_metadata, create_async_call)
                
                return await _load_tools_async(session, create_tool)
    
    return _run_async(load())


def tools_from_mcp_stdio(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None
) -> List[ToolCall]:
    """
    Load tools from an MCP server via stdio (subprocess).
    
    Args:
        command: The command to run
        args: Optional command arguments
        env: Optional environment variables
        
    Returns:
        List of ToolCall objects that can be added to a Toolkit
    """
    server_params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env
    )
    
    async def load():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                def create_tool(tool_metadata: Any) -> ToolCall:
                    """Create a tool that reconnects for each call."""
                    def create_async_call(name: str):
                        async def call(**kwargs):
                            # Reconnect for each call to handle stdio lifecycle
                            async with stdio_client(server_params) as (r, w):
                                async with ClientSession(r, w) as sess:
                                    await sess.initialize()
                                    result = await sess.call_tool(name=name, arguments=kwargs)
                                    return _extract_result_content(result)
                        return call
                    
                    return _create_tool_from_metadata(tool_metadata, create_async_call)
                
                return await _load_tools_async(session, create_tool)
    
    return _run_async(load())

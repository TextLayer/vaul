"""Utilities for working with MCP servers."""

from typing import Any, List, Dict, Optional, Callable
import asyncio
import concurrent.futures
import logging
import threading

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
        return asyncio.run(coro)
    try:
        import nest_asyncio  # type: ignore
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        def runner():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(runner).result()


def _extract_tool_metadata(tool: Any) -> tuple[str, str, dict]:
    """Extract name, description, and schema from various tool formats."""
    name = getattr(tool, "name", None) or (tool.get("name", "") if isinstance(tool, dict) else "")
    if not name:
        raise ValueError("Tool must have a name")
    description = getattr(tool, "description", "") or (tool.get("description", "") if isinstance(tool, dict) else "")
    schema: dict = {}
    for attr in ("inputSchema", "input_schema", "parameters"):
        schema = getattr(tool, attr, None) or (tool.get(attr, {}) if isinstance(tool, dict) else {})
        if schema:
            break
    return name, description, schema


def _extract_result_content(result: Any) -> Any:
    """Extract content from various MCP result formats."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and content:
            item = content[0]
            return getattr(item, "text", None) or getattr(item, "data", None) or str(item)
        return str(content)
    if hasattr(result, "result"):
        return result.result
    return result


def _parse_tools_response(response: Any) -> List[Any]:
    """Parse tools from various response formats."""
    if hasattr(response, "tools"):
        return response.tools
    if isinstance(response, dict) and "tools" in response:
        return response["tools"]
    if isinstance(response, list):
        return response
    return []


def _create_tool_wrapper(
    name: str,
    description: str,
    schema: dict,
    async_call_func: Callable[..., Any],
) -> ToolCall:
    """Create a ToolCall wrapper with the given metadata and async call function."""
    def sync_wrapper(**kwargs):
        return _run_async(async_call_func(**kwargs))
    sync_wrapper.__name__ = name
    sync_wrapper.__doc__ = description
    tool = tool_call(sync_wrapper)
    tool.tool_call_schema = {
        "name": name,
        "description": description,
        "parameters": schema or {},
    }
    return tool


def _create_tool_from_metadata(
    tool_metadata: Any,
    create_async_call: Callable[[str], Callable[..., Any]],
) -> ToolCall:
    name, description, schema = _extract_tool_metadata(tool_metadata)
    async_call = create_async_call(name)
    return _create_tool_wrapper(name, description, schema, async_call)


async def _load_tools_async(
    session: ClientSession,
    create_tool: Callable[[Any], ToolCall],
) -> List[ToolCall]:
    """Load tools from an MCP session asynchronously."""
    response = await session.list_tools()
    tools_data = _parse_tools_response(response)
    tools: List[ToolCall] = []
    for item in tools_data:
        try:
            tools.append(create_tool(item))
        except Exception as e:
            logger.warning(f"Failed to create tool: {e}")
    return tools


_persistent_pools: dict[str, "_PersistentSSEPool"] = {}


class _PersistentSSEPool:
    def __init__(self, url: str, headers: dict[str, str]):
        self.url = url
        self.headers = headers
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._loop_thread, daemon=True)
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._sse_ctx = None
        self._sess_ctx = None
        self.session: Any = None
        self._lock: Optional[asyncio.Lock] = None
        self.thread.start()
        self._ready.wait()

    def _loop_thread(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._open())
        self.loop.run_forever()

    async def _open(self):
        self._sse_ctx = sse_client(self.url, headers=self.headers)
        read, write = await self._sse_ctx.__aenter__()
        self._sess_ctx = ClientSession(read, write)
        self.session = await self._sess_ctx.__aenter__()
        await self.session.initialize()
        self._lock = asyncio.Lock()
        self._ready.set()

    async def _call_tool(self, name: str, arguments: dict):
        assert self.session is not None and self._lock is not None
        async with self._lock:
            return await self.session.call_tool(name=name, arguments=arguments)

    async def _list_tools(self):
        assert self.session is not None and self._lock is not None
        async with self._lock:
            return await self.session.list_tools()

    def call_tool_async(self, name: str, arguments: dict):
        coro = self._call_tool(name, arguments)
        cfut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return asyncio.wrap_future(cfut)

    def list_tools_async(self):
        coro = self._list_tools()
        cfut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return asyncio.wrap_future(cfut)

    def close(self):
        if self._closed.is_set():
            return

        async def _shutdown():
            try:
                if self._sess_ctx is not None:
                    await self._sess_ctx.__aexit__(None, None, None)
                if self._sse_ctx is not None:
                    await self._sse_ctx.__aexit__(None, None, None)
            finally:
                self.loop.stop()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)
        fut.result()
        self.thread.join(timeout=2)
        self._closed.set()


def _get_pool(url: str, headers: dict[str, str]) -> _PersistentSSEPool:
    pool = _persistent_pools.get(url)
    if pool is None:
        pool = _PersistentSSEPool(url, headers)
        _persistent_pools[url] = pool
    return pool


def close_mcp_url(url: str) -> None:
    pool = _persistent_pools.pop(url, None)
    if pool:
        pool.close()


def close_all_mcp_urls() -> None:
    urls = list(_persistent_pools.keys())
    for url in urls:
        close_mcp_url(url)


def tools_from_mcp(session: ClientSession) -> List[ToolCall]:
    def create_tool(tool_metadata: Any) -> ToolCall:
        def create_async_call(name: str):
            async def call(**kwargs):
                result = await session.call_tool(name=name, arguments=kwargs)
                return _extract_result_content(result)
            return call
        return _create_tool_from_metadata(tool_metadata, create_async_call)
    return _run_async(_load_tools_async(session, create_tool))


def tools_from_mcp_url(url: str, headers: Dict[str, str] | None = None) -> List[ToolCall]:
    hdrs = dict(headers or {})
    pool = _get_pool(url, hdrs)

    async def load():
        resp = await pool.list_tools_async()
        tools_data = _parse_tools_response(resp)

        def create_tool(tool_md: Any) -> ToolCall:
            def create_async_call(name: str):
                async def call(**kwargs):
                    result = await pool.call_tool_async(name, kwargs)
                    return _extract_result_content(result)
                return call
            return _create_tool_from_metadata(tool_md, create_async_call)

        out: List[ToolCall] = []
        for item in tools_data:
            try:
                out.append(create_tool(item))
            except Exception as e:
                logger.warning(f"Failed to create tool: {e}")
        return out

    return _run_async(load())


def tools_from_mcp_stdio(
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[ToolCall]:
    server_params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
    )

    async def load():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                def create_tool(tool_metadata: Any) -> ToolCall:
                    def create_async_call(name: str):
                        async def call(**kwargs):
                            async with stdio_client(server_params) as (r, w):
                                async with ClientSession(r, w) as sess:
                                    await sess.initialize()
                                    result = await sess.call_tool(name=name, arguments=kwargs)
                                    return _extract_result_content(result)
                        return call

                    return _create_tool_from_metadata(tool_metadata, create_async_call)

                return await _load_tools_async(session, create_tool)

    return _run_async(load())

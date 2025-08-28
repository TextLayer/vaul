import asyncio
import time

import pytest

from tests.utils.assertion import contains, is_equal, is_not_none, is_true
from vaul import Toolkit, tool_call


@tool_call
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool_call
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def test_toolkit_initialization():
    """Test toolkit initialization."""
    toolkit = Toolkit()
    is_equal(len(toolkit), 0)
    is_equal(toolkit.has_tools(), False)
    is_equal(toolkit.tools, {})
    is_equal(toolkit.tool_names, [])


def test_add_tool():
    """Test adding a single tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    is_equal(len(toolkit), 1)
    is_true(toolkit.has_tools())
    contains(toolkit.tool_names, "add_numbers")
    contains(toolkit.tools, "add_numbers")


def test_add_tools():
    """Test adding multiple tools."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)
    is_equal(len(toolkit), 2)
    is_equal(set(toolkit.tool_names), {"add_numbers", "multiply_numbers"})


def test_remove_tool():
    """Test removing a tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    is_true(toolkit.remove("add_numbers"))
    is_equal(len(toolkit), 0)
    is_equal(toolkit.has_tools(), False)
    is_equal(toolkit.remove("non_existent"), False)


def test_get_tool():
    """Test getting a tool by name."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    tool = toolkit.get_tool("add_numbers")
    is_not_none(tool)
    is_equal(tool.func.__name__, "add_numbers")
    is_equal(toolkit.get_tool("non_existent"), None)


def test_run_tool():
    """Test running a tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    result = toolkit.run_tool("add_numbers", {"a": 5, "b": 3})
    is_equal(result, 8)
    with pytest.raises(ValueError):
        toolkit.run_tool("non_existent", {})


def test_tool_schemas():
    """Test getting tool schemas."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)
    schemas = toolkit.tool_schemas()
    is_equal(len(schemas), 2)
    for schema in schemas:
        is_equal(schema["type"], "function")
        contains(schema, "function")
        tool_schema = schema["function"]
        contains({"add_numbers", "multiply_numbers"}, tool_schema["name"])
        contains(tool_schema, "parameters")
        contains(tool_schema["parameters"], "properties")
        contains(tool_schema["parameters"]["properties"], "a")
        contains(tool_schema["parameters"]["properties"], "b")


def test_clear():
    """Test clearing all tools."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)
    is_equal(len(toolkit), 2)
    toolkit.clear()
    is_equal(len(toolkit), 0)
    is_equal(toolkit.has_tools(), False)


def test_duplicate_tool():
    """Test adding duplicate tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    with pytest.raises(ValueError):
        toolkit.add(add_numbers)


def test_invalid_tool():
    """Test adding invalid tool."""
    toolkit = Toolkit()
    with pytest.raises(TypeError):
        toolkit.add("not_a_tool")


def test_run_tool_with_invalid_arguments():
    """Test running a tool with invalid arguments."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    result = toolkit.run_tool("add_numbers", {"a": "not_an_int", "b": 3})
    is_true(isinstance(result, str))
    contains(result.lower(), "validation error")
    contains(result.lower(), "input should be a valid integer")


@tool_call
async def async_add_numbers(a: int, b: int) -> int:
    """Asynchronously add two numbers."""
    await asyncio.sleep(0.01)
    return a + b


@tool_call(concurrent=True)
def concurrent_sleep_tool(duration: float) -> dict:
    """Tool that sleeps for given duration with concurrent execution."""
    time.sleep(duration)
    return {"slept": duration, "concurrent": True}


@tool_call(concurrent=True)
async def async_concurrent_tool(duration: float) -> dict:
    """Async tool with concurrent execution."""
    await asyncio.sleep(duration)
    return {"slept": duration, "async": True, "concurrent": True}


@tool_call(retry=True, raise_for_exception=True, max_timeout=2, max_backoff=1)
async def flaky_async_tool(should_fail: bool) -> dict:
    """Tool that can fail for retry testing."""
    if should_fail:
        raise ValueError("Intentional failure")
    return {"success": True}


async def test_async_run_tool_basic():
    """Test basic async_run_tool functionality."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": 5, "b": 3})
    is_equal(result, 8)


async def test_async_run_tool_with_async_function():
    """Test async_run_tool with async function."""
    toolkit = Toolkit()
    toolkit.add(async_add_numbers)

    result = await toolkit.async_run_tool("async_add_numbers", {"a": 10, "b": 20})
    is_equal(result, 30)


async def test_async_run_tool_nonexistent_tool():
    """Test async_run_tool with nonexistent tool."""
    toolkit = Toolkit()

    with pytest.raises(ValueError, match="Tool 'nonexistent' not found in registry"):
        await toolkit.async_run_tool("nonexistent", {})


async def test_async_run_tool_with_kwargs():
    """Test async_run_tool with additional kwargs."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": 5}, b=7)
    is_equal(result, 12)


async def test_async_run_tool_kwargs_override_arguments():
    """Test that kwargs override arguments dict."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": 5, "b": 3}, b=7)
    is_equal(result, 12)


async def test_async_run_tool_with_concurrent_sync_function():
    """Test async_run_tool with concurrent sync function."""
    toolkit = Toolkit()
    toolkit.add(concurrent_sleep_tool)

    start_time = time.time()
    tasks = [
        toolkit.async_run_tool("concurrent_sleep_tool", {"duration": 0.1})
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    is_equal(len(results), 3)
    for result in results:
        is_equal(result["slept"], 0.1)
        is_equal(result["concurrent"], True)

    total_time = end_time - start_time
    assert total_time < 0.25, (
        f"Expected concurrent execution < 0.25s, got {total_time}s"
    )


async def test_async_run_tool_with_concurrent_async_function():
    """Test async_run_tool with concurrent async function."""
    toolkit = Toolkit()
    toolkit.add(async_concurrent_tool)

    start_time = time.time()
    tasks = [
        toolkit.async_run_tool("async_concurrent_tool", {"duration": 0.1})
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    is_equal(len(results), 3)
    for result in results:
        is_equal(result["slept"], 0.1)
        is_equal(result["async"], True)
        is_equal(result["concurrent"], True)

    total_time = end_time - start_time
    assert total_time < 0.25, (
        f"Expected concurrent execution < 0.25s, got {total_time}s"
    )


async def test_async_run_tool_with_retry_success():
    """Test async_run_tool with retry functionality - success case."""
    toolkit = Toolkit()
    toolkit.add(flaky_async_tool)

    result = await toolkit.async_run_tool("flaky_async_tool", {"should_fail": False})
    is_equal(result["success"], True)


async def test_async_run_tool_with_retry_failure():
    """Test async_run_tool with retry functionality - failure case."""
    toolkit = Toolkit()
    toolkit.add(flaky_async_tool)

    with pytest.raises(ValueError, match="Intentional failure"):
        await toolkit.async_run_tool("flaky_async_tool", {"should_fail": True})


async def test_async_run_tool_invalid_arguments():
    """Test async_run_tool with invalid arguments."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": "not_an_int", "b": 3})
    is_true(isinstance(result, str))
    contains(result.lower(), "validation error")


async def test_async_run_tool_multiple_tools():
    """Test async_run_tool with multiple different tools."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers, async_add_numbers)

    tasks = [
        toolkit.async_run_tool("add_numbers", {"a": 1, "b": 2}),
        toolkit.async_run_tool("multiply_numbers", {"a": 3, "b": 4}),
        toolkit.async_run_tool("async_add_numbers", {"a": 5, "b": 6}),
    ]

    results = await asyncio.gather(*tasks)

    is_equal(results[0], 3)
    is_equal(results[1], 12)
    is_equal(results[2], 11)


async def test_async_run_tool_empty_arguments():
    """Test async_run_tool with empty arguments dict."""

    @tool_call
    def no_args_tool() -> str:
        return "success"

    toolkit = Toolkit()
    toolkit.add(no_args_tool)

    result = await toolkit.async_run_tool("no_args_tool", {})
    is_equal(result, "success")


async def test_async_run_tool_performance():
    """Test async_run_tool performance with many concurrent calls."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    start_time = time.time()
    tasks = [
        toolkit.async_run_tool("add_numbers", {"a": i, "b": i + 1}) for i in range(100)
    ]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    is_equal(len(results), 100)
    for i, result in enumerate(results):
        is_equal(result, i + (i + 1))

    total_time = end_time - start_time
    assert total_time < 1.0, f"Expected < 1s for 100 calls, got {total_time}s"

import pytest
import asyncio
from vaul import Toolkit, tool_call
from tests.utils.assertion import is_equal, is_true, is_not_none, contains


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


@pytest.mark.asyncio
async def test_run_tool_async_basic():
    """Test basic async tool execution."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": 10, "b": 5})
    is_equal(result, 15)


@pytest.mark.asyncio
async def test_run_tool_async_with_kwargs():
    """Test async tool execution with additional kwargs."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": 7}, b=13)
    is_equal(result, 20)

    result = await toolkit.async_run_tool("add_numbers", {"a": 1, "b": 2}, b=8)
    is_equal(result, 9)


@pytest.mark.asyncio
async def test_run_tool_async_with_multiplication():
    """Test async execution with different tool."""
    toolkit = Toolkit()
    toolkit.add(multiply_numbers)

    result = await toolkit.async_run_tool("multiply_numbers", {"a": 6, "b": 7})
    is_equal(result, 42)


@pytest.mark.asyncio
async def test_run_tool_async_nonexistent_tool():
    """Test async execution with nonexistent tool."""
    toolkit = Toolkit()

    with pytest.raises(ValueError, match="Tool 'nonexistent' not found in registry"):
        await toolkit.async_run_tool("nonexistent", {})


@pytest.mark.asyncio
async def test_run_tool_async_with_retry_tool():
    """Test async execution with a retry-enabled tool."""
    @tool_call(retry=True, raise_for_exception=True, max_timeout=1.0, max_backoff=0.1)
    def retry_tool(x: int) -> int:
        """Tool with retry capability."""
        return x * 2

    toolkit = Toolkit()
    toolkit.add(retry_tool)

    result = await toolkit.async_run_tool("retry_tool", {"x": 21})
    is_equal(result, 42)


@pytest.mark.asyncio
async def test_run_tool_async_with_concurrent_tool():
    """Test async execution with a concurrent-enabled tool."""
    import time

    @tool_call(concurrent=True)
    def concurrent_tool(delay_ms: int) -> dict:
        """Tool that simulates work with concurrent execution."""
        time.sleep(delay_ms / 1000.0)
        return {"processed": True, "delay": delay_ms}

    toolkit = Toolkit()
    toolkit.add(concurrent_tool)

    start_time = asyncio.get_event_loop().time()
    result = await toolkit.async_run_tool("concurrent_tool", {"delay_ms": 50})
    elapsed = asyncio.get_event_loop().time() - start_time

    is_equal(result["processed"], True)
    is_equal(result["delay"], 50)
    is_true(elapsed >= 0.05)


@pytest.mark.asyncio
async def test_run_tool_async_multiple_concurrent():
    """Test running multiple async tools concurrently."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)

    tasks = [
        toolkit.async_run_tool("add_numbers", {"a": 1, "b": 2}),
        toolkit.async_run_tool("multiply_numbers", {"a": 3, "b": 4}),
        toolkit.async_run_tool("add_numbers", {"a": 5, "b": 6}),
    ]

    results = await asyncio.gather(*tasks)
    is_equal(results, [3, 12, 11])


@pytest.mark.asyncio
async def test_run_tool_async_with_actual_async_tool():
    """Test async execution with an actual async function tool."""
    @tool_call
    async def async_computation(x: int, y: int) -> dict:
        """Async tool that performs computation."""
        await asyncio.sleep(0.01)
        return {"sum": x + y, "product": x * y}

    toolkit = Toolkit()
    toolkit.add(async_computation)

    result = await toolkit.async_run_tool("async_computation", {"x": 4, "y": 5})
    is_equal(result["sum"], 9)
    is_equal(result["product"], 20)


@pytest.mark.asyncio
async def test_run_tool_async_error_handling():
    """Test error handling in async tool execution."""
    @tool_call
    def error_tool(should_fail: bool) -> str:
        """Tool that can fail on demand."""
        if should_fail:
            raise ValueError("Tool failed as requested")
        return "success"

    toolkit = Toolkit()
    toolkit.add(error_tool)

    result = await toolkit.async_run_tool("error_tool", {"should_fail": False})
    is_equal(result, "success")

    result = await toolkit.async_run_tool("error_tool", {"should_fail": True})
    is_true(isinstance(result, str))
    is_equal(result, "Tool failed as requested")


@pytest.mark.asyncio
async def test_run_tool_async_error_handling_with_raise():
    """Test error handling with raise_for_exception enabled."""
    @tool_call(raise_for_exception=True)
    def error_raise_tool(should_fail: bool) -> str:
        """Tool that raises exceptions."""
        if should_fail:
            raise ValueError("Tool failed with raise")
        return "success"

    toolkit = Toolkit()
    toolkit.add(error_raise_tool)

    result = await toolkit.async_run_tool("error_raise_tool", {"should_fail": False})
    is_equal(result, "success")

    with pytest.raises(ValueError, match="Tool failed with raise"):
        await toolkit.async_run_tool("error_raise_tool", {"should_fail": True})


@pytest.mark.asyncio
async def test_run_tool_async_validation_errors():
    """Test validation error handling in async execution."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)

    result = await toolkit.async_run_tool("add_numbers", {"a": "invalid", "b": 5})
    is_true(isinstance(result, str))
    contains(result.lower(), "validation error")


def test_run_tool_async_method_exists():
    """Test that run_tool_async method exists on Toolkit."""
    toolkit = Toolkit()
    is_true(hasattr(toolkit, "run_tool_async"))
    is_true(callable(getattr(toolkit, "run_tool_async")))

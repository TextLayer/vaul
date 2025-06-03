import pytest
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
    contains(result.lower(), "can only concatenate str")

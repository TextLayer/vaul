import pytest
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
    assert len(toolkit) == 0
    assert not toolkit.has_tools()
    assert toolkit.tools == {}
    assert toolkit.tool_names == []


def test_add_tool():
    """Test adding a single tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    assert len(toolkit) == 1
    assert toolkit.has_tools()
    assert "add_numbers" in toolkit.tool_names
    assert "add_numbers" in toolkit.tools


def test_add_tools():
    """Test adding multiple tools."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)
    assert len(toolkit) == 2
    assert set(toolkit.tool_names) == {"add_numbers", "multiply_numbers"}


def test_remove_tool():
    """Test removing a tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    assert toolkit.remove("add_numbers")
    assert len(toolkit) == 0
    assert not toolkit.has_tools()
    # Test removing non-existent tool
    assert not toolkit.remove("non_existent")


def test_get_tool():
    """Test getting a tool by name."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    tool = toolkit.get_tool("add_numbers")
    assert tool is not None
    assert tool.func.__name__ == "add_numbers"
    assert toolkit.get_tool("non_existent") is None


def test_run_tool():
    """Test running a tool."""
    toolkit = Toolkit()
    toolkit.add(add_numbers)
    result = toolkit.run_tool("add_numbers", {"a": 5, "b": 3})
    assert result == 8
    with pytest.raises(ValueError):
        toolkit.run_tool("non_existent", {})


def test_tool_schemas():
    """Test getting tool schemas."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)
    schemas = toolkit.tool_schemas()
    assert len(schemas) == 2
    for schema in schemas:
        assert schema["type"] == "function"
        assert "function" in schema
        tool_schema = schema["function"]
        assert tool_schema["name"] in {"add_numbers", "multiply_numbers"}
        assert "parameters" in tool_schema
        assert "properties" in tool_schema["parameters"]
        assert "a" in tool_schema["parameters"]["properties"]
        assert "b" in tool_schema["parameters"]["properties"]


def test_clear():
    """Test clearing all tools."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, multiply_numbers)
    assert len(toolkit) == 2
    toolkit.clear()
    assert len(toolkit) == 0
    assert not toolkit.has_tools()


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
    assert isinstance(result, str)
    assert "can only concatenate str" in result.lower()

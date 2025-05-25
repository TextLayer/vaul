from vaul.models import BaseTool


def test_base_tool_creation():
    """Test creating a BaseTool instance with extra fields."""
    tool = BaseTool(name="test_tool", description="A test tool", extra_field="value")
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.extra_field == "value"


def test_base_tool_dict_conversion():
    """Test converting BaseTool to dictionary."""
    tool = BaseTool(name="test_tool", description="A test tool")
    tool_dict = tool.model_dump()
    assert tool_dict["name"] == "test_tool"
    assert tool_dict["description"] == "A test tool"


def test_base_tool_json_conversion():
    """Test converting BaseTool to JSON."""
    tool = BaseTool(name="test_tool", description="A test tool")
    tool_json = tool.model_dump_json()
    assert "test_tool" in tool_json
    assert "A test tool" in tool_json

from vaul.models import BaseTool
from tests.utils.assertion import is_equal, contains


def test_base_tool_creation():
    """Test creating a BaseTool instance with extra fields."""
    tool = BaseTool(name="test_tool", description="A test tool", extra_field="value")
    is_equal(tool.name, "test_tool")
    is_equal(tool.description, "A test tool")
    is_equal(tool.extra_field, "value")


def test_base_tool_dict_conversion():
    """Test converting BaseTool to dictionary."""
    tool = BaseTool(name="test_tool", description="A test tool")
    tool_dict = tool.model_dump()
    is_equal(tool_dict["name"], "test_tool")
    is_equal(tool_dict["description"], "A test tool")


def test_base_tool_json_conversion():
    """Test converting BaseTool to JSON."""
    tool = BaseTool(name="test_tool", description="A test tool")
    tool_json = tool.model_dump_json()
    contains(tool_json, "test_tool")
    contains(tool_json, "A test tool")

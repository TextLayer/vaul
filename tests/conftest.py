import pytest


@pytest.fixture
def sample_tool_dict():
    """Fixture providing a sample tool dictionary for testing."""
    return {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {"param1": {"type": "string"}, "param2": {"type": "integer"}},
        },
    }

import pytest
from vaul import validation
from tests.utils.assertion import is_true, is_false


def test_validate_tool_call_success():
    """Test successful tool call validation."""
    message = {"tool_calls": [{"function": {"name": "test_function"}}]}
    schema = {"name": "test_function"}
    is_true(validation.validate_tool_call(message, schema))


def test_validate_tool_call_no_tool_calls():
    """Test validation when no tool calls are present."""
    message = {"some_other_key": "value"}
    schema = {"name": "test_function"}
    with pytest.raises(AssertionError, match="No tool call detected"):
        validation.validate_tool_call(message, schema)


def test_validate_tool_call_wrong_function():
    """Test validation when function name doesn't match."""
    message = {"tool_calls": [{"function": {"name": "wrong_function"}}]}
    schema = {"name": "test_function"}
    with pytest.raises(AssertionError, match="Function name does not match"):
        validation.validate_tool_call(message, schema)


def test_validate_tool_call_no_throw():
    """Test validation when throw_error is False."""
    message = {"some_other_key": "value"}
    schema = {"name": "test_function"}
    is_false(validation.validate_tool_call(message, schema, throw_error=False))

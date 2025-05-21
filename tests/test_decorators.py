import json
from typing import Optional
import pytest
from pydantic import BaseModel
from vaul.decorators import StructuredOutput, tool_call


class TestOutput(StructuredOutput):
    """Test output class."""

    value: int
    text: str
    optional: Optional[bool] = None


def test_structured_output_schema():
    """Test schema generation for StructuredOutput."""
    schema = TestOutput.tool_call_schema
    assert schema["name"] == "TestOutput"
    assert "description" in schema
    assert "parameters" in schema
    assert "value" in schema["parameters"]["properties"]
    assert "text" in schema["parameters"]["properties"]
    assert "optional" in schema["parameters"]["properties"]
    assert sorted(schema["parameters"]["required"]) == ["text", "value"]


def test_structured_output_validation():
    """Test validation of structured output."""
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "TestOutput",
                    "arguments": json.dumps({"value": 42, "text": "test"}),
                }
            }
        ]
    }

    # Mock completion object
    class MockCompletion:
        class Choice:
            class Message(BaseModel):
                def model_dump(self, exclude_unset=True):
                    return message

            message = Message()

        choices = [Choice()]

    output = TestOutput.from_response(MockCompletion())
    assert output.value == 42
    assert output.text == "test"
    assert output.optional is None


def test_structured_output_validation_error():
    """Test validation errors in structured output."""
    message = {"no_tool_calls": {}}
    with pytest.raises(AssertionError, match="No tool call detected"):
        TestOutput._validate_tool_call(message)


def test_structured_output_wrong_function_name():
    """Test validation when function name doesn't match."""
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "WrongName",
                    "arguments": json.dumps({"value": 42, "text": "test"}),
                }
            }
        ]
    }

    class MockCompletion:
        class Choice:
            class Message(BaseModel):
                def model_dump(self, exclude_unset=True):
                    return message

            message = Message()

        choices = [Choice()]

    with pytest.raises(AssertionError, match="Function name does not match"):
        TestOutput.from_response(MockCompletion())


def test_structured_output_no_throw():
    """Test validation without throwing errors."""
    message = {"no_tool_calls": {}}
    assert not TestOutput._validate_tool_call(message, throw_error=False)


@tool_call
def sample_function(x: int, y: str, z: Optional[float] = 1.0) -> dict:
    """Sample function for testing."""
    return {"result": x, "text": y, "optional": z}


def test_tool_call_decorator():
    """Test the tool_call decorator functionality."""
    result = sample_function(42, "test")
    assert result == {"result": 42, "text": "test", "optional": 1.0}


def test_tool_call_schema():
    """Test schema generation for tool_call decorator."""
    schema = sample_function.tool_call_schema
    assert schema["name"] == "sample_function"
    assert schema["description"] == "Sample function for testing."
    assert "parameters" in schema
    assert "x" in schema["parameters"]["properties"]
    assert "y" in schema["parameters"]["properties"]
    assert "z" in schema["parameters"]["properties"]
    assert sorted(schema["parameters"].get("required", [])) == ["x", "y"]


def test_tool_call_validation():
    """Test validation of tool call."""
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "sample_function",
                    "arguments": json.dumps({"x": 42, "y": "test"}),
                }
            }
        ]
    }

    class MockCompletion:
        class Choice:
            class Message(BaseModel):
                def model_dump(self, exclude_unset=True):
                    return message

            message = Message()

        choices = [Choice()]

    result = sample_function.from_response(MockCompletion())
    assert result == {"result": 42, "text": "test", "optional": 1.0}


def test_tool_call_validation_error():
    """Test validation errors in tool call."""
    message = {"no_tool_calls": {}}
    with pytest.raises(AssertionError, match="No tool call detected"):
        sample_function._validate_tool_call(message)


def test_tool_call_wrong_function_name():
    """Test validation when function name doesn't match."""
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "wrong_function",
                    "arguments": json.dumps({"x": 42, "y": "test"}),
                }
            }
        ]
    }

    class MockCompletion:
        class Choice:
            class Message(BaseModel):
                def model_dump(self, exclude_unset=True):
                    return message

            message = Message()

        choices = [Choice()]

    with pytest.raises(AssertionError, match="Function name does not match"):
        sample_function.from_response(MockCompletion())


def test_tool_call_no_throw():
    """Test validation without throwing errors."""
    message = {"no_tool_calls": {}}
    assert not sample_function._validate_tool_call(message, throw_error=False)


def test_tool_call_run():
    """Test the run method of tool call."""
    result = sample_function.run({"x": 42, "y": "test"})
    assert result == {"result": 42, "text": "test", "optional": 1.0}


@tool_call(raise_for_exception=True)
def error_function_raise(x: int) -> int:
    """Function that raises an error and propagates it."""
    raise ValueError("Test error")


@tool_call
def error_function(x: int) -> int:
    """Function that raises an error and returns the error message."""
    raise ValueError("Test error")


def test_tool_call_exception_handling():
    """Test exception handling in tool_call decorator (default: no raise)."""
    result = error_function(42)
    assert isinstance(result, str)
    assert result == "Test error"

    # error_function_raise should raise
    with pytest.raises(ValueError, match="Test error"):
        error_function_raise(42)


def test_tool_call_run_exception_handling():
    """Test exception handling in tool_call run method (default: no raise)."""
    result = error_function.run({"x": 42})
    assert isinstance(result, str)
    assert result == "Test error"

    # error_function_raise should raise
    with pytest.raises(ValueError, match="Test error"):
        error_function_raise.run({"x": 42})

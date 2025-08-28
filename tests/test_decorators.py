import json
import time
import asyncio
from typing import Optional
import pytest
from pydantic import BaseModel
from vaul.decorators import StructuredOutput, tool_call
from tests.utils.assertion import is_equal, is_true, is_false, contains


class TestOutput(StructuredOutput):
    """Test output class."""

    value: int
    text: str
    optional: Optional[bool] = None


def test_structured_output_schema():
    """Test schema generation for StructuredOutput."""
    schema = TestOutput.tool_call_schema
    is_equal(schema["name"], "TestOutput")
    contains(schema, "description")
    contains(schema, "parameters")
    contains(schema["parameters"]["properties"], "value")
    contains(schema["parameters"]["properties"], "text")
    contains(schema["parameters"]["properties"], "optional")
    is_equal(sorted(schema["parameters"]["required"]), ["text", "value"])


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

    class MockCompletion:
        class Choice:
            class Message(BaseModel):
                def model_dump(self, exclude_unset=True):
                    return message

            message = Message()

        choices = [Choice()]

    output = TestOutput.from_response(MockCompletion())
    is_equal(output.value, 42)
    is_equal(output.text, "test")
    is_equal(output.optional, None)


def test_structured_output_from_dict():
    """Test creating StructuredOutput from a dictionary."""
    data = {"value": 1, "text": "foo"}
    output = TestOutput.from_dict(data)
    is_equal(output.value, 1)
    is_equal(output.text, "foo")
    is_equal(output.optional, None)


@tool_call
def use_output(data: TestOutput) -> str:
    """Function using StructuredOutput as argument."""
    return f"{data.text}-{data.value}"


def test_structured_output_auto_conversion_run():
    """ToolCall.run should convert dictionaries to StructuredOutput."""
    result = use_output.run({"data": {"value": 7, "text": "bar"}})
    is_equal(result, "bar-7")


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
    result = TestOutput._validate_tool_call(message, throw_error=False)
    is_equal(result, False)


@tool_call
def sample_function(x: int, y: str, z: Optional[float] = 1.0) -> dict:
    """Sample function for testing."""
    return {"result": x, "text": y, "optional": z}


def test_tool_call_decorator():
    """Test the tool_call decorator functionality."""
    result = sample_function(42, "test")
    is_equal(result, {"result": 42, "text": "test", "optional": 1.0})


def test_tool_call_schema():
    """Test schema generation for tool_call decorator."""
    schema = sample_function.tool_call_schema
    is_equal(schema["name"], "sample_function")
    is_equal(schema["description"], "Sample function for testing.")
    contains(schema, "parameters")
    contains(schema["parameters"]["properties"], "x")
    contains(schema["parameters"]["properties"], "y")
    contains(schema["parameters"]["properties"], "z")
    is_equal(sorted(schema["parameters"].get("required", [])), ["x", "y"])


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
    is_equal(result, {"result": 42, "text": "test", "optional": 1.0})


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
    result = sample_function._validate_tool_call(message, throw_error=False)
    is_equal(result, False)


def test_tool_call_run():
    """Test the run method of tool call."""
    result = sample_function.run({"x": 42, "y": "test"})
    is_equal(result, {"result": 42, "text": "test", "optional": 1.0})


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
    is_true(isinstance(result, str))
    is_equal(result, "Test error")

    with pytest.raises(ValueError, match="Test error"):
        error_function_raise(42)


def test_tool_call_run_exception_handling():
    """Test exception handling in tool_call run method (default: no raise)."""
    result = error_function.run({"x": 42})
    is_true(isinstance(result, str))
    is_equal(result, "Test error")

    with pytest.raises(ValueError, match="Test error"):
        error_function_raise.run({"x": 42})


def test_tool_call_new_parameters():
    """Test new ToolCall decorator parameters."""
    @tool_call(retry=True, raise_for_exception=True, max_timeout=30, max_backoff=60, concurrent=True)
    def parameterized_function(x: int) -> int:
        """Function with new parameters."""
        return x * 2

    is_equal(parameterized_function.raise_for_exception, True)
    is_equal(parameterized_function.retry, True)
    is_equal(parameterized_function.concurrent, True)
    is_equal(parameterized_function._max_timeout, 30)
    is_equal(parameterized_function._max_backoff, 60)


def test_tool_call_default_timeout_values():
    """Test default timeout values when retry is enabled."""
    @tool_call(retry=True, raise_for_exception=True)
    def retry_function(x: int) -> int:
        """Function with retry enabled and default timeout values."""
        return x

    is_equal(retry_function._max_timeout, 60)
    is_equal(retry_function._max_backoff, 120)


@pytest.mark.asyncio
async def test_run_async_basic():
    """Test basic async execution."""
    @tool_call
    def async_test_function(x: int, y: str) -> dict:
        """Test function for async execution."""
        return {"result": x, "text": y}

    result = await async_test_function.async_run({"x": 42, "y": "test"})
    is_equal(result, {"result": 42, "text": "test"})


@pytest.mark.asyncio
async def test_run_async_with_actual_async_function():
    """Test async execution with an actual async function."""
    @tool_call
    async def actual_async_function(x: int) -> int:
        """Actual async function."""
        await asyncio.sleep(0.01)
        return x * 2

    result = await actual_async_function.async_run({"x": 21})
    is_equal(result, 42)


@pytest.mark.asyncio
async def test_run_async_concurrent_mode():
    """Test async execution in concurrent mode."""
    @tool_call(concurrent=True)
    def concurrent_function(x: int) -> int:
        """Function executed in concurrent mode."""
        time.sleep(0.01)
        return x * 3

    result = await concurrent_function.async_run({"x": 10})
    is_equal(result, 30)





@pytest.mark.asyncio
async def test_run_async_exception_handling():
    """Test exception handling in async execution."""
    @tool_call
    def async_error_function(x: int) -> int:
        """Function that raises an error."""
        raise ValueError("Async test error")

    result = await async_error_function.async_run({"x": 42})
    is_true(isinstance(result, str))
    is_equal(result, "Async test error")


@pytest.mark.asyncio
async def test_run_async_exception_handling_with_raise():
    """Test exception handling in async execution with raise enabled."""
    @tool_call(raise_for_exception=True)
    def async_error_raise_function(x: int) -> int:
        """Function that raises an error and propagates it."""
        raise ValueError("Async test error with raise")

    with pytest.raises(ValueError, match="Async test error with raise"):
        await async_error_raise_function.async_run({"x": 42})


def test_concurrent_parameter_behavior():
    """Test behavior of concurrent parameter in non-async context."""
    @tool_call(concurrent=True)
    def concurrent_sync_function(x: int) -> int:
        """Sync function with concurrent flag."""
        return x * 5

    result = concurrent_sync_function.run({"x": 4})
    is_equal(result, 20)





def test_schema_generation_with_new_parameters():
    """Test that schema generation is unaffected by new parameters."""
    @tool_call(retry=True, raise_for_exception=True, concurrent=True, max_timeout=10)
    def schema_test_function(a: int, b: str, c: Optional[float] = 2) -> dict:
        """Function to test schema generation with new parameters."""
        return {"a": a, "b": b, "c": c}

    schema = schema_test_function.tool_call_schema
    is_equal(schema["name"], "schema_test_function")
    contains(schema, "parameters")
    contains(schema["parameters"]["properties"], "a")
    contains(schema["parameters"]["properties"], "b")
    contains(schema["parameters"]["properties"], "c")
    is_equal(sorted(schema["parameters"].get("required", [])), ["a", "b"])


@pytest.mark.asyncio
async def test_validation_in_async_mode():
    """Test that validation still works in async mode."""
    @tool_call
    def validation_test_function(x: int, y: str) -> dict:
        """Function to test validation in async mode."""
        return {"x": x, "y": y}

    result = await validation_test_function.async_run({"x": 100, "y": "valid"})
    is_equal(result, {"x": 100, "y": "valid"})

    result = await validation_test_function.async_run({"x": "invalid", "y": "valid"})
    is_true(isinstance(result, str))
    contains(result.lower(), "validation error")

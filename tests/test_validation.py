import pytest
from vaul import validation, tool_call
from tests.utils.assertion import is_true, is_false, is_equal, contains


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


def test_schema_generation_consistency():
    """Test that schema generation is consistent with new decorator parameters."""
    @tool_call(retry=True, raise_for_exception=True, concurrent=True, max_timeout=30.0, max_backoff=60.0)
    def complex_function(x: int, y: str, z: float = 1.0) -> dict:
        """Complex function with all new parameters."""
        return {"x": x, "y": y, "z": z}

    schema = complex_function.tool_call_schema

    is_equal(schema["name"], "complex_function")
    is_equal(schema["description"], "Complex function with all new parameters.")
    contains(schema, "parameters")

    params = schema["parameters"]
    is_equal(params["type"], "object")
    contains(params["properties"], "x")
    contains(params["properties"], "y")
    contains(params["properties"], "z")
    is_equal(sorted(params["required"]), ["x", "y"])


def test_validation_error_messages_preserved():
    """Test that validation error messages are still properly formatted."""
    @tool_call
    def strict_function(count: int, name: str) -> str:
        """Function with strict validation."""
        return f"{name}: {count}"

    result = strict_function.run({"count": "not_a_number", "name": "test"})
    is_true(isinstance(result, str))
    contains(result.lower(), "validation error")
    contains(result.lower(), "input should be a valid integer")


def test_validation_with_retry_parameter_combination():
    """Test validation when retry parameter requires raise_for_exception."""
    @tool_call(retry=True, raise_for_exception=True)
    def valid_retry_function(x: int) -> int:
        return x * 2

    is_true(valid_retry_function.retry)
    is_true(valid_retry_function.raise_for_exception)

    with pytest.raises(ValueError, match="If retry is True, raise_for_exception must also be True"):
        @tool_call(retry=True, raise_for_exception=False)
        def invalid_retry_function(x: int) -> int:
            return x


def test_parameter_defaults_with_new_features():
    """Test that parameter defaults work correctly with new features."""
    @tool_call(concurrent=True)
    def default_params_function(required: str, optional: int = 42, flag: bool = True) -> dict:
        """Function with default parameters and concurrent execution."""
        return {"required": required, "optional": optional, "flag": flag}

    schema = default_params_function.tool_call_schema

    is_equal(schema["parameters"]["required"], ["required"])
    result = default_params_function.run({"required": "test"})
    is_equal(result["required"], "test")
    is_equal(result["optional"], 42)
    is_equal(result["flag"], True)


def test_complex_type_validation_preserved():
    """Test that complex type validation still works with new features."""
    from typing import List, Optional

    @tool_call(retry=True, raise_for_exception=True, concurrent=True)
    def complex_types_function(items: List[str], count: Optional[int] = None) -> dict:
        """Function with complex types."""
        return {"items": items, "count": count or len(items)}

    schema = complex_types_function.tool_call_schema

    contains(schema["parameters"]["properties"], "items")
    contains(schema["parameters"]["properties"], "count")
    is_equal(schema["parameters"]["required"], ["items"])

    result = complex_types_function.run({"items": ["a", "b", "c"]})
    is_equal(result["items"], ["a", "b", "c"])
    is_equal(result["count"], 3)


def test_edge_case_parameter_handling():
    """Test edge cases in parameter handling with new features."""
    @tool_call(max_timeout=1.0, max_backoff=2.0, concurrent=True)
    def edge_case_function(value: str = "") -> str:
        """Function with edge case parameters."""
        return value or "default"

    result = edge_case_function.run({})
    is_equal(result, "default")

    result = edge_case_function.run({"value": ""})
    is_equal(result, "default")

    result = edge_case_function.run({"value": "custom"})
    is_equal(result, "custom")


def test_timeout_parameter_validation():
    """Test validation of timeout and max_backoff parameters."""
    @tool_call(retry=True, raise_for_exception=True, max_timeout=5.0, max_backoff=10.0)
    def valid_timeout_function(x: int) -> int:
        return x

    is_equal(valid_timeout_function._max_timeout, 5.0)
    is_equal(valid_timeout_function._max_backoff, 10.0)

    @tool_call(retry=True, raise_for_exception=True)
    def default_timeout_function(x: int) -> int:
        return x

    is_equal(default_timeout_function._max_timeout, 60.0)
    is_equal(default_timeout_function._max_backoff, 120.0)

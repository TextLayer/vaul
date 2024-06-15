from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict

from pydantic import validate_arguments

from .models import BaseTool
from .utils import remove_keys_recursively
from .validation import validate_tool_call


class ToolCall(BaseTool):
    """
    Decorator to convert a function into a tool call for an LLM.
    The function will be validated using pydantic and the schema will be
    generated from the function signature.

    Example:
        ```python
        @ToolCall
        def sum(a: int, b: int) -> int:
            return a + b
        ```

    Methods:
    - __init__: Initializes the decorator with the function to wrap.
    - _generate_tool_call_schema: Generates the schema based on the function signature.
    - __call__: Makes the class instance callable, effectively wrapping the decorated function.
    - _validate_tool_call: Validates a message against the tool call schema.
    - from_response: Creates an instance from an OpenAI API response.
    - run: Runs the function with the given arguments.

    Attributes:
    - func: The function that is being decorated.
    - validate_func: A function that wraps the original function, adding Pydantic validation.
    - tool_call_schema: The generated schema for the tool call.

    **INSPIRED BY JASON LIU'S EXCELLENT OPENAI_FUNCTION_CALL, NOW INSTRUCTOR, PACKAGE**

    https://pypi.org/project/instructor/
    """

    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func
        self.validate_func = validate_arguments(func)
        self.tool_call_schema = self._generate_tool_call_schema()

    def _generate_tool_call_schema(self) -> Dict[str, Any]:
        schema = self.validate_func.model.model_json_schema()
        relevant_properties = {
            k: v
            for k, v in schema["properties"].items()
            if k not in ("v__duplicate_kwargs", "args", "kwargs")
        }
        schema["properties"] = relevant_properties

        # Update the required field to allow empty arguments
        schema["required"] = sorted(
            k for k, v in relevant_properties.items() if v.get("default", None) is None
        ) if relevant_properties else []

        schema = remove_keys_recursively(schema, "additionalProperties")
        schema = remove_keys_recursively(schema, "title")

        return {
            "name": self.func.__name__,
            "description": self.func.__doc__,
            "parameters": schema,
        }

    def __call__(self, *args, **kwargs) -> Any:
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.validate_func(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def _validate_tool_call(
        self, message: Dict[str, Any], throw_error: bool = True
    ) -> bool:
        return validate_tool_call(message, self.tool_call_schema, throw_error)

    def from_response(self, completion: Any, throw_error: bool = True) -> Any:
        import json

        message = completion.choices[0].message.model_dump(exclude_unset=True)
        if throw_error:
            assert "tool_calls" in message, "No tool call detected"
            assert (
                message["tool_calls"][0]["function"]["name"] == self.func.__name__
            ), "Function name does not match"

        return self.validate_func(
            **json.loads(
                message["tool_calls"][0]["function"]["arguments"], strict=False
            )
        )

    def run(self, arguments: Dict[str, Any]) -> Any:
        return self.func(**arguments)


def tool_call(func: Callable) -> ToolCall:
    """
    Function to apply the ToolCall decorator to a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        ToolCall: An instance of the ToolCall class.
    """
    return ToolCall(func)

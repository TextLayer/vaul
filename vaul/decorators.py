from __future__ import annotations

import inspect
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Dict, Optional

from pydantic import TypeAdapter, ValidationError, validate_call

from .models import BaseTool
from .utils import remove_keys_recursively
from .validation import validate_tool_call


class StructuredOutput(BaseTool):
    """
    StructuredOutput is a base class that standardizes structured outputs from tool calls.
    It leverages Pydantic's BaseModel for input validation and schema generation.

    Class Methods:
    - _generate_schema_parameters: Generates parameters for a structured output schema.
    - tool_call_schema: Generates the full schema of the tool call, including the name and required parameters.
    - _validate_tool_call: Validates whether the message has a 'tool_call' field and if the name matches the schema.
    - from_response: Creates an instance from an OpenAI API response.
    - run: Runs the function with the given arguments.

    Example:
    ```python
    class MyFunction(StructuredOutput):
        arg1: int
        arg2: str
    ```

    Attributes:
    Inherits attributes from Pydantic's BaseModel.
    """

    @classmethod
    def _generate_schema_parameters(cls):
        schema = cls.model_json_schema()
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        if "description" not in schema:
            schema["description"] = (
                f"Correctly extracted `{cls.__name__}` with all the required parameters with correct types"
            )

        parameters = remove_keys_recursively(parameters, "additionalProperties")
        parameters = remove_keys_recursively(parameters, "title")

        return schema, parameters

    @classmethod
    @property
    def tool_call_schema(cls):
        schema, parameters = cls._generate_schema_parameters()
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def _validate_tool_call(cls, message, throw_error=True):
        return validate_tool_call(message, cls.tool_call_schema, throw_error)

    @classmethod
    def from_response(cls, completion, throw_error=True):
        import json

        message = completion.choices[0].message.model_dump(exclude_unset=True)
        if throw_error:
            assert "tool_calls" in message, "No tool call detected"
            assert message["tool_calls"][0]["function"]["name"] == cls.__name__, (
                "Function name does not match"
            )

        return cls(
            **json.loads(
                message["tool_calls"][0]["function"]["arguments"], strict=False
            )
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an instance from a dictionary representation."""
        return cls.model_validate(data)


class ToolCall(BaseTool):
    """
        Decorator to convert a function into a tool call for an LLM.
        The function will be validated using pydantic and the schema will be
        generated from the function signature.

        Exception Handling:
        - By default, any exceptions are caught and returned as the raw exception object
        - When raise_for_exception=True, exceptions are propagated as normal
        - This behavior applies to both direct function calls and the run() method

        Example:
            ```python
            @ToolCall
            def sum(a: int, b: int) -> int:
                return a + b

            # Returns raw exception object
            result = sum(1, "invalid")  # Returns TypeError object

            # Propagates exception
            result = sum(1, "invalid", raise_for_exception=True)  # Raises TypeError
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

    def __init__(self, func: Callable, raise_for_exception: bool = False, retry: bool = False, max_timeout: Optional[int] = None, max_backoff: Optional[int] = None, concurrent: bool = False) -> None:
        """
        Initialize the ToolCall decorator.

        Args:
            func: The function to be decorated
            raise_for_exception: Raises an exception when a tool call fails, required if retry is set to True
            retry: Whether to retry a tool call execution upon failure. If set to True, raise_for_exception must also be True
            max_timeout: Maximum tool call retry timeout in seconds
            max_backoff: Maximum time to run exponential backoff on tool call failure in seconds
            concurrent: Whether to execute this tool call concurrently (e.g. in a thread or async context)

        Raises:
            ValueError: If retry is True but raise_for_exception is False.
        """
        super().__init__()

        if retry and not raise_for_exception:
            raise ValueError("If retry is True, raise_for_exception must also be True")

        self.func = func
        self.validate_func = validate_call(func)
        self.tool_call_schema = self._generate_tool_call_schema()
        self.raise_for_exception = raise_for_exception
        self.retry = retry
        self.concurrent = concurrent
        if self.retry:
            self._max_timeout = max_timeout if max_timeout is not None else 60
            self._max_backoff = max_backoff if max_backoff is not None else 120
        else:
            self._max_timeout = None
            self._max_backoff = None

    def _generate_tool_call_schema(self) -> Dict[str, Any]:
        sig = inspect.signature(self.func)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ("args", "kwargs"):
                continue

            if param.annotation is not inspect.Parameter.empty:
                try:
                    param_adapter = TypeAdapter(param.annotation)
                    param_schema = param_adapter.json_schema()
                    properties[name] = param_schema
                except (TypeError, ValueError, ValidationError):
                    properties[name] = {"type": "object"}
            else:
                properties[name] = {"type": "object"}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        schema = {
            "properties": properties,
            "required": sorted(required) if required else [],
            "type": "object",
        }
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
            try:
                return self.validate_func(*args, **kwargs)
            except Exception as e:
                if self.raise_for_exception:
                    raise
                return str(e)

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
            assert message["tool_calls"][0]["function"]["name"] == self.func.__name__, (
                "Function name does not match"
            )

        return self.validate_func(
            **json.loads(
                message["tool_calls"][0]["function"]["arguments"], strict=False
            )
        )

    def run(self, arguments: Dict[str, Any]) -> Any:
        if not self.retry:
            try:
                return self.validate_func(**arguments)
            except Exception as e:
                if self.raise_for_exception:
                    raise
                return str(e)

        start_time = time.monotonic()
        attempt = 0

        while True:
            try:
                return self.validate_func(**arguments)
            except Exception as e:
                elapsed = time.monotonic() - start_time
                if elapsed >= self._max_timeout:
                    if self.raise_for_exception:
                        raise
                    return str(e)

                backoff = min(0.1 * (2 ** attempt), self._max_backoff)
                time.sleep(backoff)
                attempt += 1

    async def async_run(self, arguments: Dict[str, Any]) -> Any:
        if not self.retry:
            return await self._async_execute(**arguments)

        start_time = time.monotonic()
        attempt = 0

        while True:
            try:
                return await self._async_execute(**arguments)
            except Exception:
                elapsed = time.monotonic() - start_time
                if elapsed >= self._max_timeout:
                    raise

                backoff = min(0.1 * (2 ** attempt), self._max_backoff)
                await asyncio.sleep(backoff)
                attempt += 1

    async def _async_execute(self, **kwargs) -> Any:
        try:
            if self.concurrent:
                loop = asyncio.get_running_loop()
                if asyncio.iscoroutinefunction(self.validate_func):
                    return await self.validate_func(**kwargs)
                with ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(executor, lambda: self.validate_func(**kwargs))
            result = self.validate_func(**kwargs)
            return await result if asyncio.iscoroutine(result) else result
        except Exception as e:
            if self.raise_for_exception:
                raise
            return str(e)


def tool_call(func: Callable = None, *, raise_for_exception: bool = False, retry: bool = False, max_timeout: Optional[int] = None, max_backoff: Optional[int] = None, concurrent: bool = False) -> ToolCall:
    """
    Function to apply the ToolCall decorator to a function.

    Args:
        func: The function to be decorated
        raise_for_exception: Whether to raise an exception when a tool call fails, required if retry is set to True
        retry: Whether to retry a tool call execution upon failure. If set to True, raise_for_exception must also be True
        max_timeout: Maximum tool call retry timeout in seconds
        max_backoff: Maximum time to run exponential backoff on tool call failure in seconds
        concurrent: Whether to execute this tool call concurrently (e.g. in a thread or async context)
    """
    if func is not None and callable(func):
        return ToolCall(func, raise_for_exception=raise_for_exception, retry=retry, max_timeout=max_timeout, max_backoff=max_backoff, concurrent=concurrent)

    def wrapper(f):
        return ToolCall(f, raise_for_exception=raise_for_exception, retry=retry, max_timeout=max_timeout, max_backoff=max_backoff, concurrent=concurrent)

    return wrapper

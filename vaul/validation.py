from typing import Any, Dict


def validate_tool_call(
    message: Dict[str, Any], schema: Dict[str, Any], throw_error: bool = True
) -> bool:
    if throw_error:
        assert "tool_calls" in message, "No tool call detected"
        assert (
            message["tool_calls"][0]["function"]["name"] == schema["name"]
        ), "Function name does not match"
    return throw_error

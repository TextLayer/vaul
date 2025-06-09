import re
from typing import Any, Dict, List, Union

import requests
import yaml

from .decorators import tool_call, ToolCall
from .utils import remove_keys_recursively


def _create_api_tool(name: str, method: str, url: str, description: str, schema: Dict[str, Any]) -> ToolCall:
    """Create a ToolCall that performs an HTTP request."""

    def api_function(**kwargs):
        local_url = url
        for param in re.findall(r"{(.*?)}", url):
            if param in kwargs:
                local_url = local_url.replace(f"{{{param}}}", str(kwargs.pop(param)))
        if method.upper() == "GET":
            response = requests.request(method, local_url, params=kwargs)
        else:
            response = requests.request(method, local_url, json=kwargs)
        try:
            return response.json()
        except Exception:
            return response.text

    api_function.__name__ = name
    api_function.__doc__ = description or ""

    tool = tool_call(api_function)
    tool.tool_call_schema = schema
    return tool


def tools_from_openapi(spec: Union[str, Dict[str, Any]]) -> List[ToolCall]:
    """Convert an OpenAPI specification to ToolCall instances."""
    if isinstance(spec, dict):
        spec_dict = spec
    else:
        if spec.strip().startswith("http://") or spec.strip().startswith("https://"):
            text = requests.get(spec).text
        else:
            text = spec
        spec_dict = yaml.safe_load(text)

    base_url = ""
    servers = spec_dict.get("servers")
    if servers and isinstance(servers, list) and servers:
        base_url = servers[0].get("url", "")

    tools: List[ToolCall] = []

    for path, path_item in spec_dict.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                continue

            name = operation.get("operationId") or f"{method}_{path.strip('/').replace('/', '_')}"
            description = operation.get("summary") or operation.get("description", "")

            properties: Dict[str, Any] = {}
            required: List[str] = []
            params = []
            params.extend(path_item.get("parameters", []))
            params.extend(operation.get("parameters", []))

            for param in params:
                if "$ref" in param:
                    continue
                schema = param.get("schema", {"type": "string"})
                properties[param["name"]] = schema
                if param.get("required"):
                    required.append(param["name"])

            if "requestBody" in operation:
                content = operation["requestBody"].get("content", {})
                if "application/json" in content:
                    body_schema = content["application/json"].get("schema", {})
                    if body_schema.get("type") == "object":
                        properties.update(body_schema.get("properties", {}))
                        required.extend(body_schema.get("required", []))
                    elif body_schema:
                        properties["data"] = body_schema
                        if operation["requestBody"].get("required"):
                            required.append("data")

            param_schema = {
                "type": "object",
                "properties": properties,
                "required": sorted(set(required)),
            }
            param_schema = remove_keys_recursively(param_schema, "title")
            param_schema = remove_keys_recursively(param_schema, "additionalProperties")

            schema = {"name": name, "description": description, "parameters": param_schema}

            tool = _create_api_tool(name, method.upper(), base_url + path, description, schema)
            tools.append(tool)

    return tools

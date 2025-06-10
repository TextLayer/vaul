import re
from typing import Any, Dict, List, Union, Optional

import requests
import yaml

from .decorators import tool_call, ToolCall
from .utils import remove_keys_recursively


def _create_api_tool(name: str, method: str, url: str, description: str, 
                     schema: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
                     params: Optional[Dict[str, str]] = None,
                     session: Optional[requests.Session] = None) -> ToolCall:
    """Create a ToolCall that performs an HTTP request."""

    def api_function(**kwargs):
        local_url = url
        # Replace path parameters
        for param in re.findall(r"{(.*?)}", url):
            if param in kwargs:
                local_url = local_url.replace(f"{{{param}}}", str(kwargs.pop(param)))
        
        # Prepare request arguments
        request_kwargs = {
            "method": method,
            "url": local_url
        }
        
        # Add headers if provided
        if headers:
            request_kwargs["headers"] = headers.copy()
        
        # Add query params
        if method.upper() == "GET":
            if params:
                request_kwargs["params"] = {**params, **kwargs}
            else:
                request_kwargs["params"] = kwargs
        else:
            request_kwargs["json"] = kwargs
            if params:
                request_kwargs["params"] = params
        
        # Use session if provided, otherwise use requests
        if session:
            response = session.request(**request_kwargs)
        else:
            response = requests.request(**request_kwargs)
            
        try:
            return response.json()
        except Exception:
            return response.text

    api_function.__name__ = name
    api_function.__doc__ = description or ""

    tool = tool_call(api_function)
    tool.tool_call_schema = schema
    return tool


def tools_from_openapi(spec: Union[str, Dict[str, Any]], 
                      headers: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, str]] = None,
                      session: Optional[requests.Session] = None,
                      operation_ids: Optional[List[str]] = None) -> List[ToolCall]:
    """
    Convert an OpenAPI specification to ToolCall instances.
    
    Args:
        spec: OpenAPI specification as a URL, file path, dict, or YAML/JSON string
        headers: Optional headers to include in all API requests
        params: Optional query parameters to include in all API requests
        session: Optional requests.Session for custom authentication
        operation_ids: Optional list of operation IDs to import (imports all if None)
        
    Returns:
        List of ToolCall instances for the OpenAPI operations
    """
    if isinstance(spec, dict):
        spec_dict = spec
    else:
        if spec.strip().startswith("http://") or spec.strip().startswith("https://"):
            # Use session if provided for fetching the spec
            if session:
                text = session.get(spec).text
            else:
                text = requests.get(spec).text
        else:
            # Try to read as file first
            try:
                with open(spec, 'r') as f:
                    text = f.read()
            except (OSError, IOError):
                # If not a file, treat as raw YAML/JSON string
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

            operation_id = operation.get("operationId")
            name = operation_id or f"{method}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
            
            # Skip if operation_ids filter is provided and this operation is not in it
            if operation_ids and operation_id not in operation_ids:
                continue
                
            description = operation.get("summary") or operation.get("description", "")

            properties: Dict[str, Any] = {}
            required: List[str] = []
            params_list = []
            params_list.extend(path_item.get("parameters", []))
            params_list.extend(operation.get("parameters", []))

            for param in params_list:
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

            tool = _create_api_tool(
                name, method.upper(), base_url + path, description, schema,
                headers=headers, params=params, session=session
            )
            tools.append(tool)

    return tools

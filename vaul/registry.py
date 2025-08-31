from typing import Dict, Any, List, Optional

import pandas as pd
from tabulate import tabulate

from vaul.decorators import ToolCall


class Toolkit:
    """
    A class for managing and organizing multiple tool calls within an application.

    The Toolkit provides a centralized registry for tool calls, allowing them to be registered,
    accessed, and executed by name. This is particularly useful for applications that need to
    manage multiple tools for AI systems like OpenAI's GPT models.

    The Toolkit uses pandas DataFrame internally for efficient storage and retrieval of tools.

    Example:
        ```python
        from vaul import Toolkit, tool_call

        # Create a toolkit
        toolkit = Toolkit()

        @tool_call
        def add_numbers(a: int, b: int) -> int:
            return a + b

        # Register the tool
        toolkit.add(add_numbers)

        # Run the tool
        result = toolkit.run_tool("add_numbers", {"a": 5, "b": 3})
        print(result)  # Output: 8
        ```
    """

    def __init__(self):
        """
        Initialize a new Toolkit with an empty registry of tools.
        """
        self._tools_df = pd.DataFrame(columns=["name", "tool", "source"])

    def add(self, tool: ToolCall, source: str = "local") -> None:
        """
        Register a tool call instance in the toolkit.

        Args:
            tool (ToolCall): The tool to register

        Raises:
            TypeError: If the provided tool is not a ToolCall instance
            ValueError: If a tool with the same name already exists in the toolkit

        Example:
            ```python
            toolkit = Toolkit()

            @tool_call
            def add_numbers(a: int, b: int) -> int:
                return a + b

            toolkit.add(add_numbers)
            ```
        """
        if not isinstance(tool, ToolCall):
            raise TypeError(f"Expected ToolCall instance, got {type(tool).__name__}")

        tool_name = tool.func.__name__

        if not self._tools_df[self._tools_df["name"] == tool_name].empty:
            raise ValueError(f"A tool with name '{tool_name}' is already registered")

        new_row = pd.DataFrame({"name": [tool_name], "tool": [tool], "source": [source]})
        self._tools_df = pd.concat([self._tools_df, new_row], ignore_index=True)

    def add_tools(self, *tools: ToolCall, source: str = "local") -> None:
        """
        Register multiple tool call instances in the toolkit at once.

        Args:
            *tools: Variable number of ToolCall instances to register

        Raises:
            TypeError: If any provided tool is not a ToolCall instance
            ValueError: If a tool with the same name already exists in the toolkit

        Example:
            ```python
            toolkit = Toolkit()

            @tool_call
            def add_numbers(a: int, b: int) -> int:
                return a + b

            @tool_call
            def subtract_numbers(a: int, b: int) -> int:
                return a - b

            @tool_call
            def multiply_numbers(a: int, b: int) -> int:
                return a * b

            # Add all tools at once
            toolkit.add_tools(add_numbers, subtract_numbers, multiply_numbers)
            ```
        """
        for tool in tools:
            self.add(tool, source=source)

    def add_openapi(self, spec: Any, headers: Optional[Dict[str, str]] = None, 
                    params: Optional[Dict[str, str]] = None, 
                    session: Optional[Any] = None,
                    operation_ids: Optional[List[str]] = None) -> None:
        """
        Add tools from an OpenAPI specification.
        
        Args:
            spec: OpenAPI specification as a URL, file path, dict, or YAML/JSON string
            headers: Optional headers to include in API requests
            params: Optional query parameters to include in API requests
            session: Optional requests.Session for custom authentication
            operation_ids: Optional list of operation IDs to import (imports all if None)
            
        Example:
            ```python
            # From URL
            toolkit.add_openapi("https://api.example.com/openapi.json")
            
            # With authentication
            toolkit.add_openapi(
                "https://api.example.com/openapi.json",
                headers={"X-API-Key": "your-key"}
            )
            
            # Filter specific operations
            toolkit.add_openapi(
                "https://api.example.com/openapi.json",
                operation_ids=["getUserById", "createUser"]
            )
            ```
        """
        from vaul.openapi import tools_from_openapi
        
        tools = tools_from_openapi(
            spec, 
            headers=headers, 
            params=params, 
            session=session,
            operation_ids=operation_ids
        )
        
        for tool in tools:
            self.add(tool, source="openapi")

    def add_mcp(self, mcp_source: Any, **kwargs) -> None:
        """
        Add tools from an MCP server.
        
        Args:
            mcp_source: Can be one of:
                - ClientSession: An existing MCP ClientSession
                - str: URL for SSE-based MCP server (e.g., "https://mcp.example.com/sse")
                - dict: Configuration for stdio-based MCP server with keys:
                    - command: Command to run (required)
                    - args: List of arguments (optional)
                    - env: Environment variables (optional)
            **kwargs: Additional arguments passed to the MCP loader functions
        
        Example:
            ```python
            toolkit = Toolkit()
            
            # Add from URL-based MCP server
            toolkit.add_mcp("https://actions.zapier.com/mcp/sse")
            
            # Add from stdio-based MCP server
            toolkit.add_mcp({
                "command": "python3",
                "args": ["./mcp_server.py"]
            })
            
            # Add from existing session
            async with ClientSession(read, write) as session:
                await session.initialize()
                toolkit.add_mcp(session)
            ```
        """
        from vaul.mcp import tools_from_mcp, tools_from_mcp_url, tools_from_mcp_stdio
        
        # Determine the type of MCP source and load tools accordingly
        if hasattr(mcp_source, 'call_tool'):
            # It's a ClientSession
            tools = tools_from_mcp(mcp_source)
        elif isinstance(mcp_source, str):
            # It's a URL
            tools = tools_from_mcp_url(mcp_source, **kwargs)
        elif isinstance(mcp_source, dict):
            # It's a stdio configuration
            self._validate_stdio_config(mcp_source)
            tools = tools_from_mcp_stdio(
                command=mcp_source['command'],
                args=mcp_source.get('args', []),
                env=mcp_source.get('env')
            )
        else:
            raise TypeError(
                f"Unsupported MCP source type: {type(mcp_source).__name__}. "
                "Expected ClientSession, str (URL), or dict (stdio config)."
            )
        
        # Add all loaded tools to the registry
        for tool in tools:
            self.add(tool, source="mcp")
    
    def _validate_stdio_config(self, config: dict) -> None:
        """Validate stdio configuration dictionary."""
        if 'command' not in config:
            raise ValueError("MCP stdio configuration must include 'command'")
        
        if not isinstance(config['command'], str):
            raise TypeError("'command' must be a string")
        
        if 'args' in config and not isinstance(config['args'], list):
            raise TypeError("'args' must be a list of strings")
        
        if 'env' in config and not isinstance(config['env'], dict):
            raise TypeError("'env' must be a dictionary")

    def remove(self, name: str) -> bool:
        """
        Unregister a tool by name from the toolkit.

        Args:
            name (str): The name of the tool to unregister

        Returns:
            bool: True if the tool was successfully unregistered, False if not found

        Example:
            ```python
            toolkit = Toolkit()

            @tool_call
            def add_numbers(a: int, b: int) -> int:
                return a + b

            toolkit.add(add_numbers)

            # Later, remove the tool
            toolkit.remove("add_numbers")  # Returns: True
            ```
        """
        if self._tools_df[self._tools_df["name"] == name].empty:
            return False

        self._tools_df = self._tools_df[self._tools_df["name"] != name].reset_index(
            drop=True
        )
        return True

    @property
    def tools(self) -> Dict[str, ToolCall]:
        """
        Retrieve a dictionary of all registered tools.

        Returns:
            Dict[str, ToolCall]: A dictionary mapping tool names to their ToolCall instances

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            all_tools = toolkit.tools
            for name, tool in all_tools.items():
                print(f"Tool: {name}")
            ```
        """
        if self._tools_df.empty:
            return {}

        return dict(zip(self._tools_df["name"], self._tools_df["tool"]))

    @property
    def tool_names(self) -> List[str]:
        """
        Retrieve a list of all registered tool names.

        Returns:
            List[str]: A list of all registered tool names

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            names = toolkit.tool_names
            print(f"Available tools: {', '.join(names)}")
            ```
        """
        return self._tools_df["name"].tolist()

    def get_tool(self, name: str) -> Optional[ToolCall]:
        """
        Get a specific tool by name.

        Args:
            name (str): The name of the tool to retrieve

        Returns:
            Optional[ToolCall]: The tool if found, None otherwise

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            add_tool = toolkit.get_tool("add_numbers")
            if add_tool:
                result = add_tool.run({"a": 5, "b": 3})
                print(result)  # Output: 8
            ```
        """
        tool_row = self._tools_df[self._tools_df["name"] == name]

        if tool_row.empty:
            return None

        return tool_row.iloc[0]["tool"]

    def run_tool(self, name: str, arguments: Dict[str, Any], **kwargs) -> Any:
        """
        Execute the tool corresponding to the given name with the provided arguments.

        Args:
            name (str): The name of the tool to run
            arguments (Dict[str, Any]): Arguments to pass to the tool
            **kwargs: Additional keyword arguments to pass to the tool

        Raises:
            ValueError: If the tool is not found in the registry

        Returns:
            Any: The result of running the tool

        Example:
            ```python
            toolkit = Toolkit()

            @tool_call
            def add_numbers(a: int, b: int) -> int:
                return a + b

            toolkit.add(add_numbers)

            result = toolkit.run_tool("add_numbers", {"a": 5, "b": 3})
            print(result)  # Output: 8
            ```
        """
        tool = self.get_tool(name)

        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry.")

        merged_arguments = {**arguments, **kwargs}

        return tool.run(merged_arguments)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of tool call schemas for all registered tools.

        This is particularly useful when integrating with AI models like OpenAI's GPT,
        which require tool schemas in a specific format.

        Returns:
            List[Dict[str, Any]]: List of tool call schemas in the format required by OpenAI

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            # Get schemas for all tools to pass to OpenAI
            schemas = toolkit.tool_schemas()

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[...],
                tools=schemas
            )
            ```
        """
        if self._tools_df.empty:
            return []

        return [
            {"type": "function", "function": tool["tool"].tool_call_schema}
            for _, tool in self._tools_df.iterrows()
        ]

    def clear(self) -> None:
        """
        Clear all registered tools from the toolkit.

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            # Remove all tools
            toolkit.clear()
            print(len(toolkit))  # Output: 0
            ```
        """
        self._tools_df = pd.DataFrame(columns=["name", "tool", "source"])

    def has_tools(self) -> bool:
        """
        Check if the toolkit has any registered tools.

        Returns:
            bool: True if the toolkit has registered tools, False otherwise

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            print(toolkit.has_tools())  # Output: True
            ```
        """
        return not self._tools_df.empty

    def __len__(self) -> int:
        """
        Get the number of registered tools.

        Returns:
            int: Number of registered tools

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            print(f"Number of tools: {len(toolkit)}")
            ```
        """
        return len(self._tools_df)

    def to_markdown(self) -> str:
        """
        Convert the toolkit's tools to a Markdown table.

        This method generates a Markdown formatted table of all tools registered in the toolkit,
        with columns for the tool name, its description, and when to use the tool.

        The docstring of each tool is parsed to extract:
        - Line starting with "Desc:" as the description
        - Line starting with "Usage:" as the usage guidance

        If "Desc:" is not found, the first line of the docstring is used as the description.

        Returns:
            str: A Markdown formatted table of all tools

        Example:
            ```python
            toolkit = Toolkit()
            # ... add some tools ...

            markdown_table = toolkit.to_markdown()
            print(markdown_table)
            ```
        """
        if self._tools_df.empty:
            return "No tools registered."

        data = []
        for _, row in self._tools_df.iterrows():
            tool_info = self._extract_tool_info(row["tool"], row["name"])
            data.append(tool_info)

        headers = ["Tool", "Description", "When to Use"]
        table = tabulate(data, headers=headers, tablefmt="pipe")

        return f"### Tools\n{table}"

    def _extract_tool_info(self, tool: ToolCall, name: str) -> List[str]:
        """
        Extract formatted information from a tool for documentation.

        Args:
            tool: The ToolCall instance
            name: The name of the tool

        Returns:
            A list with [formatted_name, description, usage_guidance]
        """
        description = "No description available"
        usage = ""

        if tool.func.__doc__:
            docstring = tool.func.__doc__.strip()
            doc_lines = docstring.split("\n")

            if doc_lines:
                description = doc_lines[0].strip()

            current_section = None
            section_content = []

            for i, line in enumerate(doc_lines):
                line = line.strip()

                if line.lower().startswith("desc:"):
                    if current_section == "usage" and section_content:
                        usage = " ".join(section_content)

                    current_section = "desc"
                    section_content = [line[len("desc:") :].strip()]

                elif line.lower().startswith("usage:"):
                    if current_section == "desc" and section_content:
                        description = " ".join(section_content)

                    current_section = "usage"
                    section_content = [line[len("usage:") :].strip()]

                elif (
                    current_section
                    and i > 0
                    and line
                    and not line.lower().startswith(("desc:", "usage:"))
                ):
                    section_content.append(line)

            if current_section == "desc" and section_content:
                description = " ".join(section_content)
            elif current_section == "usage" and section_content:
                usage = " ".join(section_content)

        formatted_name = f"`{name}`"

        return [formatted_name, description, usage]

    async def async_run_tool(self, name: str, arguments: Dict[str, Any], **kwargs) -> Any:
        """
        Asynchronously execute the tool corresponding to the given name with the provided arguments.

        Args:
            name (str): The name of the tool to run
            arguments (Dict[str, Any]): Arguments to pass to the tool
            **kwargs: Additional keyword arguments to pass to the tool

        Raises:
            ValueError: If the tool is not found in the registry

        Returns:
            Any: The result of running the tool

        Example:
            ```python
            toolkit = Toolkit()

            @tool_call(concurrent=True)
            def add_numbers(a: int, b: int) -> int:
                return a + b

            toolkit.add(add_numbers)

            result = toolkit.async_run_tool("add_numbers", {"a": 5, "b": 3})
            print(result)  # Output: 8
            ```
        """
        tool = self.get_tool(name)

        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry.")

        merged_arguments = {**arguments, **kwargs}

        return await tool.async_run(merged_arguments)

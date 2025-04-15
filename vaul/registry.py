from typing import Dict, Any, List, Optional
import pandas as pd
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
        # Create a DataFrame with columns for the tool name and the tool object
        self._tools_df = pd.DataFrame(columns=['name', 'tool'])

    def add(self, tool: ToolCall) -> None:
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
        
        # Check if tool with same name already exists
        if not self._tools_df[self._tools_df['name'] == tool_name].empty:
            raise ValueError(f"A tool with name '{tool_name}' is already registered")
            
        # Add the tool to the DataFrame
        new_row = pd.DataFrame({'name': [tool_name], 'tool': [tool]})
        self._tools_df = pd.concat([self._tools_df, new_row], ignore_index=True)

    def add_tools(self, *tools: ToolCall) -> None:
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
            self.add(tool)

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
        if self._tools_df[self._tools_df['name'] == name].empty:
            return False
            
        self._tools_df = self._tools_df[self._tools_df['name'] != name].reset_index(drop=True)
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
        # Handle empty DataFrame case
        if self._tools_df.empty:
            return {}
            
        # Convert the DataFrame back to a dictionary for compatibility
        return dict(zip(self._tools_df['name'], self._tools_df['tool']))
    
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
        return self._tools_df['name'].tolist()
        
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
        tool_row = self._tools_df[self._tools_df['name'] == name]
        
        if tool_row.empty:
            return None
            
        return tool_row.iloc[0]['tool']

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
        # Find the tool by name in the DataFrame
        tool = self.get_tool(name)
        
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry.")
        
        # Merge the arguments and kwargs into a single dictionary
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
            
        return [{
            "type": "function",
            "function": tool['tool'].tool_call_schema
        } for _, tool in self._tools_df.iterrows()]
        
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
        self._tools_df = pd.DataFrame(columns=['name', 'tool'])

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


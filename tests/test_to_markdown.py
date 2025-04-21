from vaul import Toolkit, tool_call


@tool_call
def add_numbers(a: int, b: int) -> int:
    """Add two numbers
    
    Desc: Adds two numbers together.
    Usage: When you need to calculate the sum of two numbers.
    """
    return a + b


@tool_call
def subtract_numbers(a: int, b: int) -> int:
    """Subtract numbers
    
    Desc: Subtracts the second number from the first.
    Usage: When you need to calculate the difference between two numbers.
    """
    return a - b


@tool_call
def multiply_numbers(a: int, b: int) -> int:
    """Multiply numbers
    
    Desc: Multiplies two numbers together.
    """
    return a * b


def test_to_markdown_empty_toolkit():
    """Test that to_markdown returns a message for an empty toolkit."""
    toolkit = Toolkit()
    markdown = toolkit.to_markdown()
    assert markdown == "No tools registered."


def test_to_markdown_with_tools():
    """Test that to_markdown returns a properly formatted markdown table."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, subtract_numbers, multiply_numbers)
    
    markdown = toolkit.to_markdown()
    
    # Verify the table contains the expected tool names and descriptions
    assert "`add_numbers`" in markdown
    assert "`subtract_numbers`" in markdown
    assert "`multiply_numbers`" in markdown
    assert "Adds two numbers together." in markdown
    assert "Subtracts the second number from the first." in markdown
    assert "Multiplies two numbers together." in markdown
    # Verify it has the usage information
    assert "When you need to calculate the sum of two numbers." in markdown
    assert "When you need to calculate the difference between two numbers." in markdown
    # Verify it's in markdown table format with the expected headers
    assert "### Tools" in markdown
    assert "Tool" in markdown
    assert "Description" in markdown
    assert "When to Use" in markdown
    assert "|" in markdown


def test_to_markdown_with_no_docstring():
    """Test that to_markdown handles tools without docstrings."""
    toolkit = Toolkit()
    
    @tool_call
    def no_docs(x: int) -> int:
        return x
    
    toolkit.add(no_docs)
    markdown = toolkit.to_markdown()
    
    assert "`no_docs`" in markdown
    assert "No description available" in markdown
    # Verify header components are present
    assert "Tool" in markdown
    assert "Description" in markdown
    assert "When to Use" in markdown


def test_to_markdown_with_only_first_line_docstring():
    """Test that to_markdown handles tools with just a simple docstring."""
    toolkit = Toolkit()
    
    @tool_call
    def simple_doc(x: int) -> int:
        """Just a simple one-line docstring."""
        return x
    
    toolkit.add(simple_doc)
    markdown = toolkit.to_markdown()
    
    assert "`simple_doc`" in markdown
    assert "Just a simple one-line docstring." in markdown 


def test_to_markdown_with_multiline_docs():
    """Test that to_markdown handles multiline descriptions and usage guidance."""
    toolkit = Toolkit()
    
    @tool_call
    def multiline_doc(query: str) -> dict:
        """Search function
        
        Desc: Performs a comprehensive search 
        across multiple databases and knowledge sources
        to find relevant information.
        
        Usage: Use this tool when you need to find 
        specific information about a topic or answer 
        complex questions that require searching through data.
        """
        return {"results": [f"Result for {query}"]}
    
    toolkit.add(multiline_doc)
    markdown = toolkit.to_markdown()
    
    # Verify the tool name is present
    assert "`multiline_doc`" in markdown
    
    # Check that the multiline description is combined into a single line
    assert "Performs a comprehensive search across multiple databases and knowledge sources to find relevant information." in markdown
    
    # Check that the multiline usage is combined into a single line  
    assert "Use this tool when you need to find specific information about a topic or answer complex questions that require searching through data." in markdown 
from vaul import Toolkit, tool_call
from tests.utils.assertion import is_equal, contains


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
    is_equal(markdown, "No tools registered.")


def test_to_markdown_with_tools():
    """Test that to_markdown returns a properly formatted markdown table."""
    toolkit = Toolkit()
    toolkit.add_tools(add_numbers, subtract_numbers, multiply_numbers)

    markdown = toolkit.to_markdown()

    contains(markdown, "`add_numbers`")
    contains(markdown, "`subtract_numbers`")
    contains(markdown, "`multiply_numbers`")
    contains(markdown, "Adds two numbers together.")
    contains(markdown, "Subtracts the second number from the first.")
    contains(markdown, "Multiplies two numbers together.")
    contains(markdown, "When you need to calculate the sum of two numbers.")
    contains(markdown, "When you need to calculate the difference between two numbers.")
    contains(markdown, "### Tools")
    contains(markdown, "Tool")
    contains(markdown, "Description")
    contains(markdown, "When to Use")
    contains(markdown, "|")


def test_to_markdown_with_no_docstring():
    """Test that to_markdown handles tools without docstrings."""
    toolkit = Toolkit()

    @tool_call
    def no_docs(x: int) -> int:
        return x

    toolkit.add(no_docs)
    markdown = toolkit.to_markdown()

    contains(markdown, "`no_docs`")
    contains(markdown, "No description available")
    contains(markdown, "Tool")
    contains(markdown, "Description")
    contains(markdown, "When to Use")


def test_to_markdown_with_only_first_line_docstring():
    """Test that to_markdown handles tools with just a simple docstring."""
    toolkit = Toolkit()

    @tool_call
    def simple_doc(x: int) -> int:
        """Just a simple one-line docstring."""
        return x

    toolkit.add(simple_doc)
    markdown = toolkit.to_markdown()

    contains(markdown, "`simple_doc`")
    contains(markdown, "Just a simple one-line docstring.")


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

    contains(markdown, "`multiline_doc`")

    contains(
        markdown,
        "Performs a comprehensive search across multiple databases and knowledge sources to find relevant information.",
    )

    contains(
        markdown,
        "Use this tool when you need to find specific information about a topic or answer complex questions that require searching through data.",
    )

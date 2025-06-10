"""Test suite for MCP tools integration."""

import pytest
from unittest.mock import Mock, patch

from vaul import Toolkit
from vaul.mcp import (
    _extract_tool_metadata,
    _extract_result_content,
    _parse_tools_response,
    tools_from_mcp,
    tools_from_mcp_url,
    tools_from_mcp_stdio,
)


class FakeSession:
    """Fake MCP session for testing."""
    
    async def list_tools(self):
        class Tool:
            name = "echo"
            description = "Echo message"
            inputSchema = {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }

        class Response:
            tools = [Tool()]

        return Response()

    async def call_tool(self, name, arguments):
        return {name: arguments.get("message")}


class TestMCPIntegration:
    """Test MCP integration with Toolkit."""
    
    def test_add_mcp_tools_and_run(self):
        """Test adding MCP tools from session and running them."""
        toolkit = Toolkit()
        session = FakeSession()

        toolkit.add_mcp(session)

        assert "echo" in toolkit.tool_names
        assert toolkit._tools_df.loc[0, "source"] == "mcp"

        result = toolkit.run_tool("echo", {"message": "hi"})
        assert result["echo"] == "hi"
    
    def test_add_mcp_from_url(self):
        """Test adding MCP tools from URL."""
        toolkit = Toolkit()
        
        with patch('vaul.mcp.tools_from_mcp_url') as mock_tools:
            # Create a proper ToolCall mock
            from vaul.decorators import ToolCall
            mock_tool = Mock(spec=ToolCall)
            mock_tool.func = Mock()
            mock_tool.func.__name__ = "url_tool"
            mock_tools.return_value = [mock_tool]
            
            toolkit.add_mcp("https://example.com/mcp")
            
            assert len(toolkit) == 1
            mock_tools.assert_called_once_with("https://example.com/mcp")
    
    def test_add_mcp_from_stdio(self):
        """Test adding MCP tools from stdio configuration."""
        toolkit = Toolkit()
        
        with patch('vaul.mcp.tools_from_mcp_stdio') as mock_tools:
            # Create a proper ToolCall mock
            from vaul.decorators import ToolCall
            mock_tool = Mock(spec=ToolCall)
            mock_tool.func = Mock()
            mock_tool.func.__name__ = "stdio_tool"
            mock_tools.return_value = [mock_tool]
            
            toolkit.add_mcp({
                "command": "python",
                "args": ["server.py"],
                "env": {"KEY": "value"}
            })
            
            assert len(toolkit) == 1
            mock_tools.assert_called_once_with(command="python", args=["server.py"], env={"KEY": "value"})
    
    def test_add_mcp_invalid_stdio_config(self):
        """Test error handling for invalid stdio configuration."""
        toolkit = Toolkit()
        
        # Missing command
        with pytest.raises(ValueError, match="must include 'command'"):
            toolkit.add_mcp({})
        
        # Invalid command type
        with pytest.raises(TypeError, match="'command' must be a string"):
            toolkit.add_mcp({"command": 123})
        
        # Invalid args type
        with pytest.raises(TypeError, match="'args' must be a list"):
            toolkit.add_mcp({"command": "python", "args": "not_a_list"})
        
        # Invalid env type
        with pytest.raises(TypeError, match="'env' must be a dictionary"):
            toolkit.add_mcp({"command": "python", "env": "not_a_dict"})
    
    def test_add_mcp_unsupported_type(self):
        """Test error handling for unsupported MCP source types."""
        toolkit = Toolkit()
        
        with pytest.raises(TypeError, match="Unsupported MCP source type"):
            toolkit.add_mcp(123)


class TestMCPHelpers:
    """Test MCP helper functions."""
    
    def test_extract_tool_metadata_variations(self):
        """Test extracting metadata from various formats."""
        # Object with attributes
        tool1 = Mock()
        tool1.name = "tool1"
        tool1.description = "Description 1"
        tool1.inputSchema = {"type": "object"}
        
        name, desc, schema = _extract_tool_metadata(tool1)
        assert name == "tool1"
        assert desc == "Description 1"
        assert schema == {"type": "object"}
        
        # Dictionary format
        tool2 = {
            "name": "tool2",
            "description": "Description 2",
            "parameters": {"type": "string"}
        }
        
        name, desc, schema = _extract_tool_metadata(tool2)
        assert name == "tool2"
        assert desc == "Description 2"
        assert schema == {"type": "string"}
        
        # Mixed format with getattr fallback
        tool3 = Mock()
        tool3.name = None
        tool3.get = lambda x, default=None: {"name": "tool3"}.get(x, default)
        
        name, desc, schema = _extract_tool_metadata(tool3)
        assert name == "tool3"
    
    def test_extract_result_content_variations(self):
        """Test extracting content from various result formats."""
        # Content list with text
        result1 = Mock()
        result1.content = [Mock(text="Hello", data=None)]
        assert _extract_result_content(result1) == "Hello"
        
        # Content list with data
        result2 = Mock()
        result2.content = [Mock(text=None, data={"key": "value"})]
        assert _extract_result_content(result2) == {"key": "value"}
        
        # Direct content string
        result3 = Mock()
        result3.content = "Direct content"
        assert _extract_result_content(result3) == "Direct content"
        
        # Result attribute
        result4 = Mock(spec=['result'])
        result4.result = "Result value"
        assert _extract_result_content(result4) == "Result value"
        
        # Plain value
        assert _extract_result_content("plain") == "plain"
    
    def test_parse_tools_response_variations(self):
        """Test parsing tools from various response formats."""
        # Object with tools attribute
        resp1 = Mock()
        resp1.tools = ["tool1", "tool2"]
        assert _parse_tools_response(resp1) == ["tool1", "tool2"]
        
        # Dictionary with tools key
        resp2 = {"tools": ["tool3"]}
        assert _parse_tools_response(resp2) == ["tool3"]
        
        # Direct list
        resp3 = ["tool4", "tool5"]
        assert _parse_tools_response(resp3) == ["tool4", "tool5"]
        
        # Invalid formats
        assert _parse_tools_response(None) == []
        assert _parse_tools_response("invalid") == []


class TestMCPToolsCreation:
    """Test MCP tool creation functions."""
    
    @patch('vaul.mcp._run_async')
    def test_tools_from_mcp_session(self, mock_run_async):
        """Test creating tools from an MCP session."""
        session = Mock()
        mock_tools = [Mock(), Mock()]
        mock_run_async.return_value = mock_tools
        
        result = tools_from_mcp(session)
        
        assert result == mock_tools
        assert mock_run_async.called
    
    @patch('vaul.mcp._run_async')
    @patch('vaul.mcp.sse_client')
    def test_tools_from_mcp_url_with_headers(self, mock_sse_client, mock_run_async):
        """Test creating tools from URL with headers."""
        mock_tools = [Mock()]
        mock_run_async.return_value = mock_tools
        
        result = tools_from_mcp_url("https://example.com", {"Auth": "Bearer token"})
        
        assert result == mock_tools
        assert mock_run_async.called
    
    @patch('vaul.mcp._run_async')
    @patch('vaul.mcp.stdio_client')
    def test_tools_from_mcp_stdio_with_env(self, mock_stdio_client, mock_run_async):
        """Test creating tools from stdio with environment variables."""
        mock_tools = [Mock()]
        mock_run_async.return_value = mock_tools
        
        result = tools_from_mcp_stdio("node", ["server.js"], {"NODE_ENV": "test"})
        
        assert result == mock_tools
        assert mock_run_async.called


class TestMCPResultExtraction:
    """Test result extraction from MCP responses."""
    
    def test_complex_content_extraction(self):
        """Test extracting content from complex nested structures."""
        # Nested content with multiple items
        result = Mock()
        item1 = Mock()
        item1.text = None
        item1.data = None
        item2 = Mock()
        item2.text = "Second item"
        result.content = [item1, item2]
        
        # Should extract from first item even if it needs string conversion
        extracted = _extract_result_content(result)
        assert isinstance(extracted, str)
    
    def test_empty_content_list(self):
        """Test handling empty content list."""
        result = Mock()
        result.content = []
        
        extracted = _extract_result_content(result)
        assert extracted == "[]"  # String representation of empty list

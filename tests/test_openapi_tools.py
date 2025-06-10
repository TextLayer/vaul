from unittest.mock import patch, Mock

from vaul import Toolkit, tool_call

OPENAPI_SPEC = """
openapi: 3.0.0
info:
  title: Echo API
  version: 1.0.0
servers:
  - url: http://example.com
paths:
  /echo:
    post:
      operationId: echo
      summary: Echo message
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
              required:
                - message
      responses:
        '200':
          description: ok
  /users/{userId}:
    get:
      operationId: getUser
      summary: Get user by ID
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: ok
"""

def test_add_openapi_tools_and_run():
    toolkit = Toolkit()
    toolkit.add_openapi(OPENAPI_SPEC)
    assert "echo" in toolkit.tool_names
    tool = toolkit.get_tool("echo")
    assert tool is not None
    assert toolkit._tools_df.loc[0, "source"] == "openapi"
    schema = tool.tool_call_schema
    assert schema["name"] == "echo"
    assert "message" in schema["parameters"]["properties"]

    with patch("vaul.openapi.requests.request") as mock_request:
        mock_resp = Mock()
        mock_resp.json.return_value = {"echo": "hi"}
        mock_request.return_value = mock_resp
        result = toolkit.run_tool("echo", {"message": "hi"})
        mock_request.assert_called_once()
        assert result["echo"] == "hi"


def test_local_tool_source_column():
    toolkit = Toolkit()

    @tool_call
    def add(a: int, b: int) -> int:
        """Add numbers"""
        return a + b

    toolkit.add(add)
    assert toolkit._tools_df.loc[0, "source"] == "local"


def test_openapi_with_headers():
    """Test adding OpenAPI tools with custom headers."""
    toolkit = Toolkit()
    headers = {"X-API-Key": "test-key", "Authorization": "Bearer token"}
    
    toolkit.add_openapi(OPENAPI_SPEC, headers=headers)
    assert "echo" in toolkit.tool_names
    
    # Test that headers are passed to requests
    with patch("vaul.openapi.requests.request") as mock_request:
        mock_resp = Mock()
        mock_resp.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_resp
        
        toolkit.run_tool("echo", {"message": "test"})
        
        # Verify headers were included in the request
        call_args = mock_request.call_args
        assert call_args.kwargs["headers"] == headers


def test_openapi_with_params():
    """Test adding OpenAPI tools with query parameters."""
    toolkit = Toolkit()
    params = {"api_key": "12345", "version": "v2"}
    
    toolkit.add_openapi(OPENAPI_SPEC, params=params)
    
    with patch("vaul.openapi.requests.request") as mock_request:
        mock_resp = Mock()
        mock_resp.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_resp
        
        # Test GET request with params
        toolkit.run_tool("getUser", {"userId": "123"})
        
        call_args = mock_request.call_args
        # For GET requests, params should be merged
        assert call_args.kwargs["params"] == params


def test_openapi_with_session():
    """Test adding OpenAPI tools with a custom session."""
    toolkit = Toolkit()
    
    # Create a mock session
    mock_session = Mock()
    mock_session.request.return_value.json.return_value = {"result": "ok"}
    
    toolkit.add_openapi(OPENAPI_SPEC, session=mock_session)
    
    # Run a tool and verify session was used
    toolkit.run_tool("echo", {"message": "test"})
    
    # Verify session.request was called instead of requests.request
    mock_session.request.assert_called_once()
    call_args = mock_session.request.call_args
    assert call_args.kwargs["method"] == "POST"
    assert call_args.kwargs["json"] == {"message": "test"}


def test_openapi_operation_filtering():
    """Test filtering operations by operationId."""
    toolkit = Toolkit()
    
    # Add only specific operations
    toolkit.add_openapi(OPENAPI_SPEC, operation_ids=["echo"])
    
    # Should only have the echo operation
    assert "echo" in toolkit.tool_names
    assert "getUser" not in toolkit.tool_names
    assert len(toolkit.tool_names) == 1
    
    # Test with multiple operations
    toolkit2 = Toolkit()
    toolkit2.add_openapi(OPENAPI_SPEC, operation_ids=["echo", "getUser"])
    assert len(toolkit2.tool_names) == 2
    
    # Test with non-existent operation
    toolkit3 = Toolkit()
    toolkit3.add_openapi(OPENAPI_SPEC, operation_ids=["nonExistent"])
    assert len(toolkit3.tool_names) == 0


def test_openapi_file_loading():
    """Test loading OpenAPI spec from file."""
    import tempfile
    import os
    
    # Create a temporary file with the spec
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(OPENAPI_SPEC)
        temp_file = f.name
    
    try:
        toolkit = Toolkit()
        toolkit.add_openapi(temp_file)
        assert "echo" in toolkit.tool_names
        assert "getUser" in toolkit.tool_names
    finally:
        os.unlink(temp_file)


def test_openapi_path_parameter_replacement():
    """Test that path parameters are correctly replaced."""
    toolkit = Toolkit()
    toolkit.add_openapi(OPENAPI_SPEC)
    
    with patch("vaul.openapi.requests.request") as mock_request:
        mock_resp = Mock()
        mock_resp.json.return_value = {"id": "123", "name": "John"}
        mock_request.return_value = mock_resp
        
        toolkit.run_tool("getUser", {"userId": "123"})
        
        # Verify the URL had the parameter replaced
        call_args = mock_request.call_args
        assert call_args.kwargs["url"] == "http://example.com/users/123"

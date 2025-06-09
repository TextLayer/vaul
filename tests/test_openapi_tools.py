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

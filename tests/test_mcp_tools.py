from vaul import Toolkit


class FakeSession:
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


def test_add_mcp_tools_and_run():
    toolkit = Toolkit()
    session = FakeSession()

    toolkit.add_mcp(session)

    assert "echo" in toolkit.tool_names
    assert toolkit._tools_df.loc[0, "source"] == "mcp"

    result = toolkit.run_tool("echo", {"message": "hi"})
    assert result["echo"] == "hi"

from unittest.mock import Mock, patch, AsyncMock

from vaul.mcp import tools_from_mcp_url


@patch('vaul.mcp._get_pool')
def test_hidden_context_merge_precedence(mock_get_pool):
    fake_pool = Mock()
    fake_pool.list_tools_async = AsyncMock()

    tool_meta = {
        "name": "merge_tool",
        "description": "",
        "inputSchema": {},
    }

    resp = Mock()
    resp.tools = [tool_meta]
    fake_pool.list_tools_async.return_value = resp
    fake_pool.call_tool_async = AsyncMock(return_value=Mock(content=[Mock(text="ok")]))
    mock_get_pool.return_value = fake_pool

    tools = tools_from_mcp_url("http://mock-test.com", hidden_context={"a": 1, "b": 2})
    t = tools[0]
    out = t.run({"a": 9, "c": 3})
    assert out == "ok"
    fake_pool.call_tool_async.assert_awaited_with("merge_tool", {"a": 9, "b": 2, "c": 3})


@patch('vaul.mcp._get_pool')
def test_hidden_context_no_mutable_default_bleed(mock_get_pool):
    fake_pool = Mock()
    fake_pool.list_tools_async = AsyncMock()

    tool_meta = {
        "name": "t",
        "description": "",
        "inputSchema": {},
    }

    resp = Mock()
    resp.tools = [tool_meta]
    fake_pool.list_tools_async.return_value = resp
    fake_pool.call_tool_async = AsyncMock(return_value=Mock(content=[Mock(text="ok")]))
    mock_get_pool.return_value = fake_pool

    tools1 = tools_from_mcp_url("http://mock-test.com/u1")
    tools1[0].run({})
    tools2 = tools_from_mcp_url("http://mock-test.com/u2", hidden_context={"x": "y"})
    tools2[0].run({})

    calls = fake_pool.call_tool_async.await_args_list
    assert len(calls) == 2
    assert calls[0].args[0] == "t"
    assert calls[0].args[1] == {}
    assert calls[1].args[0] == "t"
    assert calls[1].args[1] == {"x": "y"}


@patch('vaul.mcp._get_pool')
def test_schema_not_exposing_hidden_context(mock_get_pool):
    fake_pool = Mock()
    fake_pool.list_tools_async = AsyncMock()

    tool_meta = {
        "name": "t",
        "description": "",
        "inputSchema": {"type": "object", "properties": {"p": {"type": "string"}}},
    }

    resp = Mock()
    resp.tools = [tool_meta]
    fake_pool.list_tools_async.return_value = resp
    fake_pool.call_tool_async = AsyncMock(return_value=Mock(content=[Mock(text="ok")]))
    mock_get_pool.return_value = fake_pool

    tools = tools_from_mcp_url("http://mock-test.com", hidden_context={"secret": "s"})
    t = tools[0]
    schema = getattr(t, "tool_call_schema", {})
    params = schema.get("parameters", {})
    assert "secret" not in str(params)

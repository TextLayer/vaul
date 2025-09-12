import pytest
from unittest.mock import Mock, patch, AsyncMock
import threading

from vaul.mcp import _persistent_pools, close_mcp_url, close_all_mcp_urls, _get_pool
from vaul.mcp import tools_from_mcp_url


class TestPersistentSSEPoolBehavior:
    @patch('vaul.mcp._get_pool')
    def test_tools_from_mcp_url_end_to_end_with_pool(self, mock_get_pool):
        fake_pool = Mock()
        fake_pool.list_tools_async = AsyncMock()
        fake_pool.call_tool_async = AsyncMock()

        class ToolMeta:
            name = "remote_echo"
            description = "Echoes"
            inputSchema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

        fake_list_resp = Mock()
        fake_list_resp.tools = [ToolMeta()]
        fake_pool.list_tools_async.return_value = fake_list_resp
        mock_get_pool.return_value = fake_pool

        tools = tools_from_mcp_url("https://example.com/mcp", {"Auth": "X"})
        assert len(tools) == 1
        tool = tools[0]
        assert tool.func.__name__ == "remote_echo"
        assert "Echoes" in (tool.func.__doc__ or "")

        content_item = Mock()
        content_item.text = "ok"
        content_item.data = None
        fake_result = Mock()
        fake_result.content = [content_item]
        fake_pool.call_tool_async.return_value = fake_result

        out = tool.run({"text": "hello"})
        assert out == "ok"

        fake_pool.list_tools_async.assert_awaited()
        fake_pool.call_tool_async.assert_awaited_with("remote_echo", {"text": "hello"})

    @patch('vaul.mcp._get_pool')
    def test_tools_from_mcp_url_handles_multiple_tools(self, mock_get_pool):
        fake_pool = Mock()
        fake_pool.list_tools_async = AsyncMock()

        class T1:
            name = "t1"
            description = ""
            inputSchema = {}

        class T2:
            name = "t2"
            description = ""
            inputSchema = {}

        resp = Mock()
        resp.tools = [T1(), T2()]
        fake_pool.list_tools_async.return_value = resp
        fake_pool.call_tool_async = AsyncMock()
        mock_get_pool.return_value = fake_pool

        tools = tools_from_mcp_url("http://server")
        names = {t.func.__name__ for t in tools}
        assert names == {"t1", "t2"}


class TestPoolLifecycle:
    def setup_method(self):
        _persistent_pools.clear()

    def test_close_mcp_url_closes_and_removes(self):
        fake_pool = Mock()
        fake_pool.close = Mock()
        _persistent_pools["http://a"] = fake_pool

        close_mcp_url("http://a")
        assert "http://a" not in _persistent_pools
        fake_pool.close.assert_called_once()

    def test_close_all_mcp_urls_closes_each(self):
        fake1 = Mock()
        fake1.close = Mock()
        fake2 = Mock()
        fake2.close = Mock()
        _persistent_pools.clear()
        _persistent_pools["u1"] = fake1
        _persistent_pools["u2"] = fake2

        close_all_mcp_urls()
        assert _persistent_pools == {}
        fake1.close.assert_called_once()
        fake2.close.assert_called_once()

    @patch('vaul.mcp._PersistentSSEPool')
    def test_get_pool_thread_safety_single_instance(self, mock_pool_cls):
        class DummyPool:
            def __init__(self, url, headers):
                self.url = url
                self.headers = headers

        mock_pool_cls.side_effect = DummyPool

        results = []

        def worker():
            p = _get_pool("http://same", {"H": "V"})
            results.append(p)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(_persistent_pools) == 1
        pools_set = {id(p) for p in results}
        assert len(pools_set) == 1


class TestMCPInitFailure:
    @patch('vaul.mcp._PersistentSSEPool', side_effect=RuntimeError("init fail"))
    def test_tools_from_mcp_url_init_failure(self, _mock_pool):
        with pytest.raises(RuntimeError, match="init fail"):
            tools_from_mcp_url("http://bad")

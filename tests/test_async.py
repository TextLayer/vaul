import asyncio
import time

from tests import BaseTest
from vaul import tool_call


@tool_call(concurrent=True)
def sync_sleep_tool(duration: float) -> dict:
    """Synchronous tool that sleeps for given duration."""
    time.sleep(duration)
    return {"slept": duration, "type": "sync"}


@tool_call(concurrent=True)
async def async_sleep_tool(duration: float) -> dict:
    """Asynchronous tool that sleeps for given duration."""
    await asyncio.sleep(duration)
    return {"slept": duration, "type": "async"}


@tool_call(concurrent=False)
async def async_no_concurrent_tool(duration: float) -> dict:
    """Asynchronous tool without concurrency."""
    await asyncio.sleep(duration)
    return {"slept": duration, "type": "async_no_concurrent"}


class TestAsyncExecution(BaseTest):
    async def test_concurrent_sync_functions_parallel_execution(self):
        """Test that concurrent=True allows sync functions to run in parallel."""
        sleep_duration = 0.2
        num_tasks = 5

        start_time = time.time()

        tasks = [
            sync_sleep_tool.async_run({"duration": sleep_duration})
            for _ in range(num_tasks)
        ]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == num_tasks
        assert all(result["slept"] == sleep_duration for result in results)
        assert all(result["type"] == "sync" for result in results)

        assert total_time < sleep_duration * 2, (
            f"Expected < {sleep_duration * 2}s, got {total_time}s"
        )
        assert total_time >= sleep_duration, (
            f"Expected >= {sleep_duration}s, got {total_time}s"
        )

    async def test_concurrent_async_functions_parallel_execution(self):
        """Test that concurrent=True works with async functions."""
        sleep_duration = 0.2
        num_tasks = 5

        start_time = time.time()

        tasks = [
            async_sleep_tool.async_run({"duration": sleep_duration})
            for _ in range(num_tasks)
        ]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == num_tasks
        assert all(result["slept"] == sleep_duration for result in results)
        assert all(result["type"] == "async" for result in results)

        assert total_time < sleep_duration * 2, (
            f"Expected < {sleep_duration * 2}s, got {total_time}s"
        )
        assert total_time >= sleep_duration, (
            f"Expected >= {sleep_duration}s, got {total_time}s"
        )

    async def test_non_concurrent_async_functions_sequential_execution(self):
        """Test that concurrent=False works correctly with async functions."""
        execution_order = []

        @tool_call(concurrent=False)
        async def ordered_async_tool(task_id: int) -> dict:
            execution_order.append(f"start_{task_id}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end_{task_id}")
            return {"task_id": task_id, "type": "async_no_concurrent"}

        tasks = [ordered_async_tool.async_run({"task_id": i}) for i in range(3)]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(result["type"] == "async_no_concurrent" for result in results)

        expected_order = ["start_0", "start_1", "start_2", "end_0", "end_1", "end_2"]
        assert execution_order == expected_order, (
            f"Expected execution order {expected_order}, got {execution_order}"
        )

    async def test_mixed_concurrent_and_non_concurrent_execution(self):
        """Test mixing concurrent and non-concurrent tools."""
        sleep_duration = 0.15

        start_time = time.time()

        tasks = [
            sync_sleep_tool.async_run({"duration": sleep_duration}),
            async_sleep_tool.async_run({"duration": sleep_duration}),
            async_no_concurrent_tool.async_run({"duration": sleep_duration}),
        ]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == 3
        assert all(result["slept"] == sleep_duration for result in results)

        assert total_time < sleep_duration * 2.5, (
            f"Expected < {sleep_duration * 2.5}s, got {total_time}s"
        )
        assert total_time >= sleep_duration, (
            f"Expected >= {sleep_duration}s, got {total_time}s"
        )

    async def test_concurrent_with_different_durations(self):
        """Test concurrent execution with varying sleep durations."""
        durations = [0.1, 0.2, 0.05, 0.15, 0.3]
        max_duration = max(durations)

        start_time = time.time()

        tasks = [
            sync_sleep_tool.async_run({"duration": duration}) for duration in durations
        ]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == len(durations)
        for i, result in enumerate(results):
            assert result["slept"] == durations[i]
            assert result["type"] == "sync"

        assert total_time < max_duration * 2, (
            f"Expected < {max_duration * 2}s, got {total_time}s"
        )
        assert total_time >= max_duration, (
            f"Expected >= {max_duration}s, got {total_time}s"
        )

    async def test_concurrent_error_isolation(self):
        """Test that errors in concurrent execution don't affect other tasks."""

        @tool_call(concurrent=True, raise_for_exception=True)
        def error_tool(should_fail: bool) -> dict:
            if should_fail:
                raise ValueError("Intentional error")
            time.sleep(0.1)
            return {"success": True}

        tasks = [
            error_tool.async_run({"should_fail": False}),
            error_tool.async_run({"should_fail": True}),
            error_tool.async_run({"should_fail": False}),
            error_tool.async_run({"should_fail": True}),
            error_tool.async_run({"should_fail": False}),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 5

        successful_results = [
            r for r in results if isinstance(r, dict) and r.get("success")
        ]
        failed_results = [r for r in results if isinstance(r, ValueError)]

        assert len(successful_results) >= 3, (
            f"Expected at least 3 successes, got {len(successful_results)}"
        )
        assert len(failed_results) >= 2, (
            f"Expected at least 2 failures, got {len(failed_results)}"
        )

        for failure in failed_results:
            assert "Intentional error" in str(failure)

    async def test_concurrent_resource_sharing(self):
        """Test that concurrent execution properly handles shared resources with synchronization."""
        shared_counter = {"value": 0}
        lock = asyncio.Lock()

        @tool_call(concurrent=True)
        async def increment_counter_safe(increment: int) -> dict:
            async with lock:
                old_value = shared_counter["value"]
                await asyncio.sleep(0.01)
                shared_counter["value"] = old_value + increment
                return {"old_value": old_value, "new_value": shared_counter["value"]}

        tasks = [increment_counter_safe.async_run({"increment": 1}) for _ in range(5)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        total_time = end_time - start_time

        assert total_time < 0.4, (
            f"Expected < 0.4s, got {total_time}s (likely ran sequentially)"
        )

        assert len(results) == 5
        assert (
            shared_counter["value"] == 5
        )
        old_values = [result["old_value"] for result in results]
        new_values = [result["new_value"] for result in results]

        assert sorted(old_values) == [0, 1, 2, 3, 4]
        assert sorted(new_values) == [1, 2, 3, 4, 5]

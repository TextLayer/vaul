import asyncio
import time

from vaul import tool_call


async def test_async_retry_success_after_failures():
    """Test that async retry succeeds after initial failures."""
    attempt_count = {"value": 0}

    @tool_call(retry=True, raise_for_exception=True, max_timeout=5, max_backoff=1)
    async def flaky_async_tool(target_attempt: int) -> dict:
        attempt_count["value"] += 1
        if attempt_count["value"] < target_attempt:
            raise ConnectionError(f"Attempt {attempt_count['value']} failed")
        return {"success": True, "attempts": attempt_count["value"]}

    start_time = time.time()
    result = await flaky_async_tool.async_run({"target_attempt": 3})
    end_time = time.time()

    assert result["success"]
    assert result["attempts"] == 3
    assert attempt_count["value"] == 3

    assert end_time - start_time >= 0.25, (
        f"Expected >= 0.25s, got {end_time - start_time}s"
    )


async def test_async_retry_exponential_backoff():
    """Test that async retry uses exponential backoff correctly."""
    attempt_times = []

    @tool_call(retry=True, raise_for_exception=True, max_timeout=10, max_backoff=2)
    async def backoff_test_tool(fail_count: int) -> dict:
        attempt_times.append(time.time())
        if len(attempt_times) <= fail_count:
            raise ValueError(f"Attempt {len(attempt_times)} failed")
        return {"success": True, "total_attempts": len(attempt_times)}

    result = await backoff_test_tool.async_run({"fail_count": 4})

    assert result["success"]
    assert result["total_attempts"] == 5
    assert len(attempt_times) == 5

    delays = [
        attempt_times[i] - attempt_times[i - 1] for i in range(1, len(attempt_times))
    ]
    assert delays[0] >= 0.08, f"First delay should be ~0.1s, got {delays[0]}s"
    assert delays[1] >= 0.15, f"Second delay should be ~0.2s, got {delays[1]}s"
    assert delays[2] >= 0.35, f"Third delay should be ~0.4s, got {delays[2]}s"
    assert delays[3] >= 0.7, f"Fourth delay should be ~0.8s, got {delays[3]}s"


async def test_async_retry_timeout_exceeded():
    """Test that async retry stops after max_timeout is exceeded."""

    @tool_call(retry=True, raise_for_exception=True, max_timeout=1, max_backoff=0.5)
    async def always_failing_tool(message: str) -> dict:
        raise RuntimeError(f"Always fails: {message}")

    start_time = time.time()

    try:
        await always_failing_tool.async_run({"message": "test"})
        assert False, "Expected RuntimeError to be raised"
    except RuntimeError as e:
        assert "Always fails: test" in str(e)

    end_time = time.time()
    total_time = end_time - start_time

    assert 0.8 <= total_time <= 2.0, f"Expected 0.8-2.0s, got {total_time}s"


async def test_async_retry_max_backoff_cap():
    """Test that async retry respects max_backoff cap."""
    attempt_times = []

    @tool_call(retry=True, raise_for_exception=True, max_timeout=8, max_backoff=1)
    async def capped_backoff_tool() -> dict:
        attempt_times.append(time.time())
        if len(attempt_times) <= 5:
            raise ValueError(f"Attempt {len(attempt_times)} failed")
        return {"success": True}

    result = await capped_backoff_tool.async_run({})

    assert result["success"]
    assert len(attempt_times) == 6

    delays = [
        attempt_times[i] - attempt_times[i - 1] for i in range(1, len(attempt_times))
    ]

    for i, delay in enumerate(delays[2:], 3):
        assert delay <= 1.2, f"Delay {i} should be capped at ~1s, got {delay}s"


async def test_async_retry_with_sync_function():
    """Test async retry with synchronous function."""
    attempt_count = {"value": 0}

    @tool_call(retry=True, raise_for_exception=True, max_timeout=3, max_backoff=0.5)
    def sync_flaky_tool(target_attempt: int) -> dict:
        attempt_count["value"] += 1
        if attempt_count["value"] < target_attempt:
            raise ValueError(f"Sync attempt {attempt_count['value']} failed")
        return {"success": True, "attempts": attempt_count["value"], "type": "sync"}

    start_time = time.time()
    result = await sync_flaky_tool.async_run({"target_attempt": 3})
    end_time = time.time()

    assert result["success"]
    assert result["attempts"] == 3
    assert result["type"] == "sync"
    assert attempt_count["value"] == 3

    assert end_time - start_time >= 0.25, (
        f"Expected >= 0.25s, got {end_time - start_time}s"
    )


async def test_async_retry_concurrent_execution():
    """Test async retry with concurrent execution."""

    @tool_call(
        retry=True,
        raise_for_exception=True,
        concurrent=True,
        max_timeout=2,
        max_backoff=0.3,
    )
    async def concurrent_retry_tool(tool_id: int, fail_count: int) -> dict:
        if not hasattr(concurrent_retry_tool, f"_attempts_{tool_id}"):
            setattr(concurrent_retry_tool, f"_attempts_{tool_id}", 0)

        attempts = getattr(concurrent_retry_tool, f"_attempts_{tool_id}")
        attempts += 1
        setattr(concurrent_retry_tool, f"_attempts_{tool_id}", attempts)

        if attempts <= fail_count:
            raise ConnectionError(f"Tool {tool_id} attempt {attempts} failed")

        return {"success": True, "tool_id": tool_id, "attempts": attempts}

    start_time = time.time()

    tasks = [
        concurrent_retry_tool.async_run({"tool_id": i, "fail_count": 2})
        for i in range(3)
    ]

    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result["success"]
        assert result["tool_id"] == i
        assert result["attempts"] == 3

    assert total_time >= 0.25, f"Expected >= 0.25s, got {total_time}s"
    assert total_time <= 1.5, (
        f"Expected <= 1.5s, got {total_time}s (might not be concurrent)"
    )


async def test_async_retry_preserves_original_exception():
    """Test that async retry preserves the original exception when timeout is exceeded."""

    class CustomError(Exception):
        pass

    @tool_call(retry=True, raise_for_exception=True, max_timeout=0.5, max_backoff=0.1)
    async def custom_error_tool(error_message: str) -> dict:
        raise CustomError(error_message)

    try:
        await custom_error_tool.async_run({"error_message": "Custom failure"})
        assert False, "Expected CustomError to be raised"
    except CustomError as e:
        assert "Custom failure" in str(e)
    except Exception as e:
        assert False, f"Expected CustomError, got {type(e).__name__}: {e}"


async def test_async_retry_no_retry_on_success():
    """Test that successful async calls don't trigger retry logic."""
    call_count = {"value": 0}

    @tool_call(retry=True, raise_for_exception=True, max_timeout=5, max_backoff=1)
    async def success_tool(data: str) -> dict:
        call_count["value"] += 1
        return {"success": True, "data": data, "call_count": call_count["value"]}

    start_time = time.time()
    result = await success_tool.async_run({"data": "test"})
    end_time = time.time()

    assert result["success"]
    assert result["data"] == "test"
    assert result["call_count"] == 1
    assert call_count["value"] == 1

    assert end_time - start_time < 0.1, f"Expected < 0.1s, got {end_time - start_time}s"


async def test_async_retry_mixed_success_failure():
    """Test async retry with mixed success and failure scenarios."""

    @tool_call(retry=True, raise_for_exception=True, max_timeout=2, max_backoff=0.5)
    async def mixed_tool(
        tool_id: int, should_fail: bool, fail_attempts: int = 2
    ) -> dict:
        if not hasattr(mixed_tool, f"_attempts_{tool_id}"):
            setattr(mixed_tool, f"_attempts_{tool_id}", 0)

        attempts = getattr(mixed_tool, f"_attempts_{tool_id}")
        attempts += 1
        setattr(mixed_tool, f"_attempts_{tool_id}", attempts)

        if should_fail and attempts <= fail_attempts:
            raise RuntimeError(f"Tool {tool_id} failing on attempt {attempts}")

        return {"success": True, "tool_id": tool_id, "attempts": attempts}

    tasks = [
        mixed_tool.async_run({"tool_id": 1, "should_fail": False}),
        mixed_tool.async_run({"tool_id": 2, "should_fail": True, "fail_attempts": 1}),
        mixed_tool.async_run({"tool_id": 3, "should_fail": True, "fail_attempts": 2}),
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 3

    assert results[0]["tool_id"] == 1
    assert results[0]["attempts"] == 1

    assert results[1]["tool_id"] == 2
    assert results[1]["attempts"] == 2

    assert results[2]["tool_id"] == 3
    assert results[2]["attempts"] == 3

    assert all(result["success"] for result in results)

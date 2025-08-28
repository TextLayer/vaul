import time
import asyncio
from typing import Optional
import pytest
from vaul.decorators import tool_call
from tests.utils.assertion import is_equal, is_true, is_false, contains


def test_tool_call_retry_validation():
    """Test validation of retry parameter requirements."""
    with pytest.raises(ValueError, match="If retry is True, raise_for_exception must also be True"):
        @tool_call(retry=True, raise_for_exception=False)
        def invalid_retry_function(x: int) -> int:
            return x

    @tool_call(retry=True, raise_for_exception=True)
    def valid_retry_function(x: int) -> int:
        return x * 2
    
    is_true(valid_retry_function.retry)
    is_true(valid_retry_function.raise_for_exception)

    @tool_call(retry=False, raise_for_exception=False)
    def no_retry_function(x: int) -> int:
        return x * 3
    
    is_false(no_retry_function.retry)
    is_false(no_retry_function.raise_for_exception)

    @tool_call(retry=False, raise_for_exception=True)
    def no_retry_but_raise_function(x: int) -> int:
        return x * 4
    
    is_false(no_retry_but_raise_function.retry)
    is_true(no_retry_but_raise_function.raise_for_exception)


def test_retry_requires_raise_for_exception_rationale():
    """Test demonstrating why retry requires raise_for_exception=True."""
    @tool_call(retry=False, raise_for_exception=False)
    def function_that_fails(x: int) -> int:
        if x < 10:
            raise ValueError("Value too low")
        return x * 2

    result = function_that_fails.run({"x": 5})
    is_equal(result, "Value too low")
    
    @tool_call(retry=False, raise_for_exception=True)
    def function_that_fails_properly(x: int) -> int:
        if x < 10:
            raise ValueError("Value too low")
        return x * 2
    
    with pytest.raises(ValueError, match="Value too low"):
        function_that_fails_properly.run({"x": 5})


def test_tool_call_function_retry_validation():
    """Test retry validation when using tool_call as a function (not decorator)."""
    
    with pytest.raises(ValueError, match="If retry is True, raise_for_exception must also be True"):
        def some_function(x: int) -> int:
            return x * 2
        
        tool_call(some_function, retry=True, raise_for_exception=False)

    def another_function(x: int) -> int:
        return x * 3
    
    valid_tool = tool_call(another_function, retry=True, raise_for_exception=True)
    is_true(valid_tool.retry)
    is_true(valid_tool.raise_for_exception)


@pytest.mark.asyncio
async def test_run_async_with_retry_success():
    """Test async execution with retry on successful function."""
    @tool_call(retry=True, raise_for_exception=True, max_timeout=5.0, max_backoff=1.0)
    def retry_success_function(x: int) -> int:
        """Function that succeeds on retry."""
        return x * 4

    result = await retry_success_function.async_run({"x": 5})
    is_equal(result, 20)


@pytest.mark.asyncio
async def test_run_async_with_retry_failure():
    """Test async execution with retry on failing function."""
    call_count = 0

    @tool_call(retry=True, raise_for_exception=True, max_timeout=0.5, max_backoff=0.1)
    def retry_fail_function(x: int) -> int:
        """Function that always fails to test retry timeout."""
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")

    start_time = time.time()
    with pytest.raises(ValueError, match="Always fails"):
        await retry_fail_function.async_run({"x": 1})

    elapsed = time.time() - start_time
    is_true(elapsed >= 0.5)
    is_true(call_count >= 2)


@pytest.mark.asyncio
async def test_run_async_with_retry_eventual_success():
    """Test async execution with retry that eventually succeeds."""
    attempt_count = 0

    @tool_call(retry=True, raise_for_exception=True, max_timeout=5.0, max_backoff=0.1)
    def retry_eventual_success(x: int) -> int:
        """Function that succeeds after a few attempts."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Temporary failure")
        return x * attempt_count

    result = await retry_eventual_success.async_run({"x": 7})
    is_equal(result, 21)
    is_equal(attempt_count, 3)


def test_run_sync_with_retry_success():
    """Test synchronous execution with retry on successful function."""
    @tool_call(retry=True, raise_for_exception=True, max_timeout=5.0, max_backoff=1.0)
    def retry_sync_success_function(x: int) -> int:
        """Function that succeeds on retry."""
        return x * 4

    result = retry_sync_success_function.run({"x": 5})
    is_equal(result, 20)


def test_run_sync_with_retry_failure():
    """Test synchronous execution with retry on failing function."""
    call_count = 0

    @tool_call(retry=True, raise_for_exception=True, max_timeout=0.5, max_backoff=0.1)
    def retry_sync_fail_function(x: int) -> int:
        """Function that always fails to test retry timeout."""
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")

    start_time = time.time()
    with pytest.raises(ValueError, match="Always fails"):
        retry_sync_fail_function.run({"x": 1})

    elapsed = time.time() - start_time
    is_true(elapsed >= 0.5)
    is_true(call_count >= 2)


def test_run_sync_with_retry_eventual_success():
    """Test synchronous execution with retry that eventually succeeds."""
    attempt_count = 0

    @tool_call(retry=True, raise_for_exception=True, max_timeout=5.0, max_backoff=0.1)
    def retry_sync_eventual_success(x: int) -> int:
        """Function that succeeds after a few attempts."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Temporary failure")
        return x * attempt_count

    result = retry_sync_eventual_success.run({"x": 7})
    is_equal(result, 21)
    is_equal(attempt_count, 3)


def test_run_sync_retry_timing():
    """Test that synchronous retry has proper exponential backoff timing."""
    attempt_count = 0

    @tool_call(retry=True, raise_for_exception=True, max_timeout=2.0, max_backoff=0.5)
    def retry_sync_timing_function(x: int) -> int:
        """Function that fails a few times to test timing."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 4:
            raise ValueError("Temporary failure")
        return x * attempt_count

    start_time = time.time()
    result = retry_sync_timing_function.run({"x": 2})
    elapsed = time.time() - start_time

    is_true(elapsed >= 0.7)
    is_equal(result, 8)
    is_equal(attempt_count, 4)


@pytest.mark.asyncio
async def test_concurrent_with_retry():
    """Test combination of concurrent and retry parameters."""
    attempt_count = 0

    @tool_call(retry=True, raise_for_exception=True, concurrent=True, max_timeout=2.0, max_backoff=0.1)
    def concurrent_retry_function(x: int) -> int:
        """Function with both concurrent and retry."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ValueError("First attempt fails")
        time.sleep(0.01)
        return x * 10

    result = await concurrent_retry_function.async_run({"x": 3})
    is_equal(result, 30)
    is_equal(attempt_count, 2)

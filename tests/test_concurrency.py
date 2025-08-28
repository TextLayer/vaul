import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaul import Toolkit, tool_call
from tests import BaseTest


@tool_call
def simple_add(a: int, b: int) -> int:
    """Simple addition for concurrency testing."""
    return a + b


@tool_call
def complex_calculation(n: int) -> int:
    """Complex calculation for concurrency testing."""
    result = 0
    for i in range(n):
        result += i**2
    return result


@tool_call
def thread_id_tool() -> dict:
    """Tool that returns thread information."""
    return {
        "thread_id": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "timestamp": time.time(),
    }


@tool_call
def stateful_counter(increment: int = 1) -> int:
    """Tool with internal state for testing race conditions."""
    if not hasattr(stateful_counter, "_counter"):
        stateful_counter._counter = 0
    stateful_counter._counter += increment
    return stateful_counter._counter


class TestConcurrencyAndThreadSafety(BaseTest):
    """Concurrency and thread safety tests."""

    def test_concurrent_tool_execution(self):
        """Test concurrent execution of tools."""
        toolkit = Toolkit()
        toolkit.add(simple_add)
        toolkit.add(complex_calculation)

        def execute_tool(tool_name, args, iterations=10):
            results = []
            for _ in range(iterations):
                result = toolkit.run_tool(tool_name, args)
                results.append(result)
            return results

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for _ in range(5):
                futures.append(
                    executor.submit(execute_tool, "simple_add", {"a": 5, "b": 3})
                )
                futures.append(
                    executor.submit(execute_tool, "complex_calculation", {"n": 100})
                )

            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)

        add_results = [r for r in all_results if r == 8]
        calc_results = [r for r in all_results if r == 328350]

        assert len(add_results) == 50
        assert len(calc_results) == 50

    def test_thread_safe_toolkit_operations(self):
        """Test thread safety of toolkit operations."""
        toolkit = Toolkit()

        def add_tools_worker(start_idx, count):
            """Worker function to add tools concurrently."""
            for i in range(start_idx, start_idx + count):

                @tool_call
                def worker_tool(x: int) -> int:
                    return x * i

                worker_tool.func.__name__ = f"worker_tool_{i}"
                try:
                    toolkit.add(worker_tool)
                except ValueError:
                    pass

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(add_tools_worker, i * 10, 10))

            for future in as_completed(futures):
                future.result()

        assert len(toolkit) > 0
        assert len(toolkit.tool_names) == len(set(toolkit.tool_names))

    def test_concurrent_schema_generation(self):
        """Test concurrent schema generation."""

        @tool_call
        def test_function(a: int, b: str, c: float) -> dict:
            """Test function for concurrent schema generation."""
            return {"result": a}

        def get_schema():
            return test_function.tool_call_schema

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_schema) for _ in range(100)]
            schemas = [future.result() for future in as_completed(futures)]

        first_schema = schemas[0]
        for schema in schemas[1:]:
            assert schema == first_schema

        assert len(schemas) == 100
        assert "parameters" in first_schema

    def test_race_condition_detection(self):
        """Test for race conditions in toolkit operations."""
        toolkit = Toolkit()
        results = []
        errors = []

        def concurrent_operations(thread_id):
            """Perform various operations concurrently."""
            try:

                @tool_call
                def race_tool(x: int) -> int:
                    return x + thread_id

                race_tool.func.__name__ = f"race_tool_{thread_id}"
                toolkit.add(race_tool)

                result = toolkit.run_tool(f"race_tool_{thread_id}", {"x": 10})
                results.append(result)

                toolkit.remove(f"race_tool_{thread_id}")

            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(20):
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print("\nRace Condition Test:")
        print(f"Successful operations: {len(results)}")
        print(f"Errors: {len(errors)}")

        assert len(results) > 10, "Too many race condition failures"
        assert len(toolkit) == 0, "Toolkit should be empty after all removals"

    def test_concurrent_tool_removal(self):
        """Test concurrent tool removal operations."""
        toolkit = Toolkit()

        for i in range(20):

            @tool_call
            def removal_tool(x: int) -> int:
                return x * i

            removal_tool.func.__name__ = f"removal_tool_{i}"
            toolkit.add(removal_tool)

        initial_count = len(toolkit)
        assert initial_count == 20

        def remove_tools_worker(tool_indices):
            """Worker function to remove tools concurrently."""
            removed_count = 0
            for i in tool_indices:
                try:
                    if toolkit.remove(f"removal_tool_{i}"):
                        removed_count += 1
                except Exception:
                    pass
            return removed_count

        with ThreadPoolExecutor(max_workers=4) as executor:
            tool_batches = [
                list(range(0, 5)),
                list(range(5, 10)),
                list(range(10, 15)),
                list(range(15, 20)),
            ]

            futures = [
                executor.submit(remove_tools_worker, batch) for batch in tool_batches
            ]

            total_removed = sum(future.result() for future in as_completed(futures))

        assert total_removed >= 15, (
            f"Expected at least 15 removals, got {total_removed}"
        )
        assert len(toolkit) <= 5, (
            f"Expected at most 5 tools remaining, got {len(toolkit)}"
        )

    def test_thread_isolation(self):
        """Test that threads don't interfere with each other's tool execution."""
        toolkit = Toolkit()
        toolkit.add(thread_id_tool)

        def execute_and_collect(iterations=10):
            """Execute tool and collect thread information."""
            thread_info = []
            for _ in range(iterations):
                result = toolkit.run_tool("thread_id_tool", {})
                thread_info.append(result)
            return thread_info

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_and_collect) for _ in range(5)]
            all_thread_info = []
            for future in as_completed(futures):
                thread_info = future.result()
                all_thread_info.extend(thread_info)

        thread_ids = {info["thread_id"] for info in all_thread_info}
        assert len(thread_ids) >= 1, (
            f"Should have at least 1 thread ID, got {len(thread_ids)}"
        )
        assert len(thread_ids) <= 5, (
            f"Should have at most 5 different thread IDs, got {len(thread_ids)}"
        )

        by_thread = {}
        for info in all_thread_info:
            thread_id = info["thread_id"]
            if thread_id not in by_thread:
                by_thread[thread_id] = []
            by_thread[thread_id].append(info)

        for thread_id, executions in by_thread.items():
            assert len(executions) >= 1, (
                f"Thread {thread_id} should have at least 1 execution"
            )
            thread_names = {exec_info["thread_name"] for exec_info in executions}
            assert len(thread_names) == 1

    def test_concurrent_toolkit_access(self):
        """Test concurrent access to toolkit properties and methods."""
        toolkit = Toolkit()

        for i in range(10):

            @tool_call
            def access_tool(x: int) -> int:
                return x + i

            access_tool.func.__name__ = f"access_tool_{i}"
            toolkit.add(access_tool)

        def concurrent_access_worker():
            """Worker that performs various toolkit operations."""
            operations_results = []

            operations_results.append(len(toolkit))
            operations_results.append(len(toolkit.tool_names))
            operations_results.append(len(toolkit.tools))
            operations_results.append(toolkit.has_tools())
            operations_results.append(len(toolkit.tool_schemas()))

            result = toolkit.run_tool("access_tool_0", {"x": 5})
            operations_results.append(result)

            return operations_results

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_access_worker) for _ in range(10)]
            all_results = [future.result() for future in as_completed(futures)]

        first_result = all_results[0]
        for result in all_results[1:]:
            assert result == first_result, "Inconsistent results from concurrent access"

    def test_deadlock_prevention(self):
        """Test that operations don't cause deadlocks."""
        toolkit = Toolkit()

        @tool_call
        def deadlock_test_tool(value: int) -> int:
            """Tool for deadlock testing."""
            time.sleep(0.001)
            return value * 2

        toolkit.add(deadlock_test_tool)

        def complex_operations(thread_id):
            """Perform complex operations that might cause deadlocks."""
            results = []

            for i in range(10):
                result = toolkit.run_tool("deadlock_test_tool", {"value": i})
                results.append(result)

                _ = len(toolkit)
                _ = toolkit.tool_names
                _ = toolkit.has_tools()

                _ = toolkit.tool_schemas()

            return results

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(complex_operations, i) for i in range(8)]

            all_results = []
            for future in as_completed(futures, timeout=30):
                results = future.result()
                all_results.extend(results)

        end_time = time.time()
        total_time = end_time - start_time

        assert total_time < 10, (
            f"Operations took too long, possible deadlock: {total_time}s"
        )
        assert len(all_results) == 80

    def test_resource_contention(self):
        """Test behavior under high resource contention."""
        toolkit = Toolkit()

        @tool_call
        def contention_tool(work_amount: int) -> dict:
            """Tool that simulates resource-intensive work."""
            result = 0
            for i in range(work_amount):
                result += i**2

            return {
                "work_amount": work_amount,
                "result": result,
                "thread_id": threading.get_ident(),
            }

        toolkit.add(contention_tool)

        def high_contention_worker(worker_id):
            """Worker that creates high resource contention."""
            results = []
            for i in range(5):
                result = toolkit.run_tool("contention_tool", {"work_amount": 1000})
                result["worker_id"] = worker_id
                results.append(result)
            return results

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(high_contention_worker, i) for i in range(20)]

            all_results = []
            for future in as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)

        assert len(all_results) == 100

        for result in all_results:
            assert result["work_amount"] == 1000
            assert result["result"] == sum(i**2 for i in range(1000))
            assert "thread_id" in result
            assert "worker_id" in result

        thread_ids = {result["thread_id"] for result in all_results}
        assert len(thread_ids) > 1, "Should use multiple threads"

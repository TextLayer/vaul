import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pytest
from vaul import Toolkit, tool_call

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@tool_call
def simple_add(a: int, b: int) -> int:
    """Simple addition for performance testing."""
    return a + b


@tool_call
def complex_calculation(n: int) -> int:
    """Complex calculation for performance testing."""
    result = 0
    for i in range(n):
        result += i ** 2
    return result


@tool_call
def memory_intensive_task(size: int) -> list:
    """Memory intensive task for testing."""
    return list(range(size))


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for vaul components."""

    def test_tool_execution_speed(self):
        """Benchmark tool execution speed."""
        toolkit = Toolkit()
        toolkit.add(simple_add)
        
        for _ in range(10):
            toolkit.run_tool("simple_add", {"a": 1, "b": 2})
        
        execution_times = []
        iterations = 1000
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = toolkit.run_tool("simple_add", {"a": 5, "b": 3})
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)
            assert result == 8
        
        avg_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        p95_time = statistics.quantiles(execution_times, n=20)[18]  # 95th percentile
        
        print("\nTool Execution Performance:")
        print(f"Average time: {avg_time:.6f}s")
        print(f"Median time: {median_time:.6f}s")
        print(f"95th percentile: {p95_time:.6f}s")
        
        assert avg_time < 0.001, f"Average execution time too slow: {avg_time}s"
        assert p95_time < 0.002, f"95th percentile too slow: {p95_time}s"

    def test_schema_generation_performance(self):
        """Benchmark schema generation performance."""
        @tool_call
        def complex_function(
            a: int, b: str, c: float, d: bool, e: list, f: dict
        ) -> dict:
            """Complex function with multiple parameter types."""
            return {"result": "test"}
        
        generation_times = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            schema = complex_function.tool_call_schema
            end_time = time.perf_counter()
            generation_times.append(end_time - start_time)
            assert "parameters" in schema
        
        avg_time = statistics.mean(generation_times)
        print("\nSchema Generation Performance:")
        print(f"Average time: {avg_time:.6f}s")
        
        assert avg_time < 0.01, f"Schema generation too slow: {avg_time}s"

    def test_toolkit_scaling_performance(self):
        """Test performance with large number of tools."""
        toolkit = Toolkit()
        
        tools = []
        for i in range(100):
            @tool_call
            def dynamic_tool(x: int) -> int:
                return x * 2
            
            dynamic_tool.func.__name__ = f"tool_{i}"
            tools.append(dynamic_tool)
        
        start_time = time.perf_counter()
        for tool in tools:
            toolkit.add(tool)
        add_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        schemas = toolkit.tool_schemas()
        schema_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        for i in range(10):
            toolkit.run_tool(f"tool_{i}", {"x": 5})
        execution_time = time.perf_counter() - start_time
        
        print("\nToolkit Scaling Performance (100 tools):")
        print(f"Adding tools: {add_time:.6f}s")
        print(f"Schema generation: {schema_time:.6f}s")
        print(f"Tool execution (10 calls): {execution_time:.6f}s")
        
        assert len(schemas) == 100
        assert add_time < 1.0, f"Adding tools too slow: {add_time}s"
        assert schema_time < 0.5, f"Schema generation too slow: {schema_time}s"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_monitoring(self):
        """Monitor memory usage during operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        toolkit = Toolkit()
        
        for i in range(50):
            @tool_call
            def memory_tool(size: int = 1000) -> list:
                return list(range(size))
            
            memory_tool.func.__name__ = f"memory_tool_{i}"
            toolkit.add(memory_tool)
        
        for i in range(10):
            toolkit.run_tool(f"memory_tool_{i}", {"size": 10000})
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print("\nMemory Usage:")
        print(f"Initial: {initial_memory:.2f} MB")
        print(f"Final: {final_memory:.2f} MB")
        print(f"Increase: {memory_increase:.2f} MB")
        
        assert memory_increase < 100, f"Memory usage too high: {memory_increase} MB"

    def test_large_dataset_handling(self):
        """Test performance with large datasets."""
        toolkit = Toolkit()
        
        @tool_call
        def large_data_processor(data: list) -> dict:
            """Process large datasets."""
            return {
                "count": len(data),
                "sum": sum(data) if all(isinstance(x, (int, float)) for x in data) else 0,
                "processed": True
            }
        
        toolkit.add(large_data_processor)
        
        dataset_sizes = [1000, 10000, 100000]
        
        for size in dataset_sizes:
            large_dataset = list(range(size))
            
            start_time = time.perf_counter()
            result = toolkit.run_tool("large_data_processor", {"data": large_dataset})
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            
            print(f"\nLarge Dataset Performance (size: {size}):")
            print(f"Processing time: {processing_time:.6f}s")
            
            assert result["count"] == size
            assert result["processed"] is True
            assert processing_time < 1.0, f"Processing too slow for size {size}: {processing_time}s"

    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        toolkit = Toolkit()
        toolkit.add(simple_add)
        
        def execute_batch(iterations=100):
            """Execute a batch of tool calls."""
            times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = toolkit.run_tool("simple_add", {"a": 5, "b": 3})
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                assert result == 8
            return times
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_batch) for _ in range(5)]
            all_times = []
            for future in as_completed(futures):
                batch_times = future.result()
                all_times.extend(batch_times)
        
        avg_concurrent_time = statistics.mean(all_times)
        print("\nConcurrent Performance:")
        print(f"Average time under load: {avg_concurrent_time:.6f}s")
        print(f"Total operations: {len(all_times)}")
        
        assert avg_concurrent_time < 0.002, f"Concurrent performance degraded: {avg_concurrent_time}s"

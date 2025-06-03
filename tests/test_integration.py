import json
from unittest.mock import Mock
from vaul import Toolkit, tool_call, StructuredOutput
from typing import List, Dict, Optional
from tests import BaseTest
from tests.utils.assertion import contains


class WeatherInfo(StructuredOutput):
    """Structured output for weather information."""

    temperature: float
    humidity: int
    conditions: str
    location: str


@tool_call
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get weather information for a city.

    Desc: Retrieves current weather conditions for the specified city.
    Usage: Use when you need current weather information for a location.
    """
    return {
        "temperature": 22.5,
        "humidity": 65,
        "conditions": "partly cloudy",
        "location": city,
    }


@tool_call
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates.

    Desc: Calculates the great circle distance between two geographic points.
    Usage: Use when you need to find the distance between two locations.
    """
    return abs(lat1 - lat2) + abs(lon1 - lon2)


@tool_call
def send_notification(message: str, recipient: str, priority: str = "normal") -> dict:
    """Send a notification message.

    Desc: Sends a notification message to the specified recipient.
    Usage: Use when you need to notify someone about something important.
    """
    return {
        "status": "sent",
        "message_id": "msg_12345",
        "recipient": recipient,
        "priority": priority,
    }


class TestOpenAIIntegration(BaseTest):
    """Integration tests with mock OpenAI responses."""

    def create_mock_completion(self, tool_name: str, arguments: dict):
        """Create a mock OpenAI completion response."""
        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        message_data = {
            "tool_calls": [
                {"function": {"name": tool_name, "arguments": json.dumps(arguments)}}
            ]
        }

        mock_message.model_dump.return_value = message_data
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]

        return mock_completion

    def test_single_tool_integration(self):
        """Test integration with a single tool call."""
        toolkit = Toolkit()
        toolkit.add(get_weather)

        mock_completion = self.create_mock_completion(
            "get_weather", {"city": "New York", "units": "celsius"}
        )

        result = get_weather.from_response(mock_completion)

        assert result["location"] == "New York"
        assert result["temperature"] == 22.5
        assert "conditions" in result

    def test_multiple_tools_workflow(self):
        """Test workflow with multiple tools."""
        toolkit = Toolkit()
        toolkit.add_tools(get_weather, calculate_distance, send_notification)

        self.create_mock_completion(
            "get_weather", {"city": "London", "units": "celsius"}
        )
        weather_result = toolkit.run_tool("get_weather", {"city": "London"})

        self.create_mock_completion(
            "calculate_distance",
            {"lat1": 51.5074, "lon1": -0.1278, "lat2": 40.7128, "lon2": -74.0060},
        )
        distance_result = toolkit.run_tool(
            "calculate_distance",
            {"lat1": 51.5074, "lon1": -0.1278, "lat2": 40.7128, "lon2": -74.0060},
        )

        self.create_mock_completion(
            "send_notification",
            {
                "message": f"Weather in London: {weather_result['conditions']}, Distance: {distance_result}km",
                "recipient": "user@example.com",
            },
        )
        notification_result = toolkit.run_tool(
            "send_notification",
            {
                "message": f"Weather in London: {weather_result['conditions']}, Distance: {distance_result}km",
                "recipient": "user@example.com",
            },
        )

        assert weather_result["location"] == "London"
        assert isinstance(distance_result, float)
        assert notification_result["status"] == "sent"
        assert notification_result["recipient"] == "user@example.com"

    def test_structured_output_integration(self):
        """Test integration with structured outputs."""
        mock_completion = self.create_mock_completion(
            "WeatherInfo",
            {
                "temperature": 25.0,
                "humidity": 70,
                "conditions": "sunny",
                "location": "Paris",
            },
        )

        weather_info = WeatherInfo.from_response(mock_completion)

        assert weather_info.temperature == 25.0
        assert weather_info.humidity == 70
        assert weather_info.conditions == "sunny"
        assert weather_info.location == "Paris"

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        toolkit = Toolkit()

        @tool_call
        def error_prone_tool(value: int) -> int:
            """Tool that may raise errors."""
            if value < 0:
                raise ValueError("Value must be positive")
            return value * 2

        toolkit.add(error_prone_tool)

        result = toolkit.run_tool("error_prone_tool", {"value": 5})
        assert result == 10

        error_result = toolkit.run_tool("error_prone_tool", {"value": -1})
        assert isinstance(error_result, str)
        assert "Value must be positive" in error_result

    def test_tool_schema_integration(self):
        """Test tool schema integration with OpenAI format."""
        toolkit = Toolkit()
        toolkit.add_tools(get_weather, calculate_distance, send_notification)

        schemas = toolkit.tool_schemas()

        assert len(schemas) == 3

        for schema in schemas:
            assert "type" in schema
            assert schema["type"] == "function"
            assert "function" in schema

            function_schema = schema["function"]
            assert "name" in function_schema
            assert "description" in function_schema
            assert "parameters" in function_schema

            parameters = function_schema["parameters"]
            assert "type" in parameters
            assert parameters["type"] == "object"
            assert "properties" in parameters

    def test_complex_argument_handling(self):
        """Test handling of complex arguments in integration."""

        @tool_call
        def complex_tool(
            simple_param: str,
            optional_param: Optional[int] = None,
            list_param: Optional[List[str]] = None,
            dict_param: Optional[Dict[str, int]] = None,
        ) -> dict:
            """Tool with complex parameter types."""
            return {
                "simple": simple_param,
                "optional": optional_param,
                "list_length": len(list_param) if list_param else 0,
                "dict_keys": list(dict_param.keys()) if dict_param else [],
            }

        toolkit = Toolkit()
        toolkit.add(complex_tool)

        result = toolkit.run_tool(
            "complex_tool",
            {
                "simple_param": "test",
                "optional_param": 42,
                "list_param": ["a", "b", "c"],
                "dict_param": {"x": 1, "y": 2},
            },
        )

        assert result["simple"] == "test"
        assert result["optional"] == 42
        assert result["list_length"] == 3
        assert set(result["dict_keys"]) == {"x", "y"}

    def test_tool_validation_integration(self):
        """Test tool validation in integration scenarios."""
        toolkit = Toolkit()

        @tool_call
        def validated_tool(positive_number: int, email_address: str) -> dict:
            """Tool with validation requirements."""
            if positive_number <= 0:
                raise ValueError("Number must be positive")
            if "@" not in email_address:
                raise ValueError("Invalid email format")
            return {"number": positive_number, "email": email_address}

        toolkit.add(validated_tool)

        result = toolkit.run_tool(
            "validated_tool",
            {"positive_number": 42, "email_address": "test@example.com"},
        )
        assert result["number"] == 42
        assert result["email"] == "test@example.com"

        invalid_result = toolkit.run_tool(
            "validated_tool",
            {"positive_number": -1, "email_address": "test@example.com"},
        )
        assert isinstance(invalid_result, str)
        assert "positive" in invalid_result.lower()


class TestEndToEndWorkflows(BaseTest):
    """End-to-end workflow tests."""

    def test_complete_assistant_workflow(self):
        """Test a complete AI assistant workflow."""
        toolkit = Toolkit()
        toolkit.add_tools(get_weather, calculate_distance, send_notification)

        workflow_steps = [
            {
                "tool": "get_weather",
                "args": {"city": "New York", "units": "fahrenheit"},
                "expected_keys": ["temperature", "humidity", "conditions", "location"],
            },
            {
                "tool": "send_notification",
                "args": {
                    "message": "Weather update for New York",
                    "recipient": "user@example.com",
                    "priority": "normal",
                },
                "expected_keys": ["status", "message_id", "recipient"],
            },
        ]

        results = []
        for step in workflow_steps:
            result = toolkit.run_tool(step["tool"], step["args"])

            for key in step["expected_keys"]:
                contains(result, key)

            results.append(result)

        assert len(results) == 2
        assert results[0]["location"] == "New York"
        assert results[1]["status"] == "sent"

    def test_error_recovery_workflow(self):
        """Test workflow with error recovery."""
        toolkit = Toolkit()

        @tool_call
        def unreliable_tool(attempt: int) -> dict:
            """Tool that fails on first attempt."""
            if attempt == 1:
                raise ConnectionError("Network timeout")
            return {"success": True, "attempt": attempt}

        @tool_call
        def fallback_tool(reason: str) -> dict:
            """Fallback tool when primary fails."""
            return {"fallback_used": True, "reason": reason}

        toolkit.add_tools(unreliable_tool, fallback_tool)

        try:
            result = toolkit.run_tool("unreliable_tool", {"attempt": 1})
            if isinstance(result, str) and "Network timeout" in result:
                result = toolkit.run_tool(
                    "fallback_tool", {"reason": "Primary tool failed"}
                )
                assert result["fallback_used"] is True
        except Exception:
            result = toolkit.run_tool("fallback_tool", {"reason": "Exception occurred"})
            assert result["fallback_used"] is True

        success_result = toolkit.run_tool("unreliable_tool", {"attempt": 2})
        assert success_result["success"] is True
        assert success_result["attempt"] == 2

    def test_complex_data_flow_workflow(self):
        """Test workflow with complex data flow between tools."""
        toolkit = Toolkit()

        @tool_call
        def data_processor(data: List[int]) -> dict:
            """Process a list of numbers."""
            return {
                "sum": sum(data),
                "average": sum(data) / len(data),
                "count": len(data),
                "processed_data": [x * 2 for x in data],
            }

        @tool_call
        def data_analyzer(processed_result: dict) -> dict:
            """Analyze processed data."""
            return {
                "analysis": "complete",
                "total_sum": processed_result["sum"],
                "data_quality": "good" if processed_result["count"] > 5 else "limited",
                "recommendation": "proceed"
                if processed_result["average"] > 10
                else "review",
            }

        toolkit.add_tools(data_processor, data_analyzer)

        input_data = [5, 10, 15, 20, 25, 30]

        processed = toolkit.run_tool("data_processor", {"data": input_data})

        analysis = toolkit.run_tool("data_analyzer", {"processed_result": processed})

        assert processed["sum"] == 105
        assert processed["average"] == 17.5
        assert processed["count"] == 6
        assert len(processed["processed_data"]) == 6

        assert analysis["analysis"] == "complete"
        assert analysis["total_sum"] == 105
        assert analysis["data_quality"] == "good"
        assert analysis["recommendation"] == "proceed"

    def test_parallel_execution_workflow(self):
        """Test workflow with parallel execution capabilities."""
        import concurrent.futures
        import time

        toolkit = Toolkit()

        @tool_call
        def parallel_task(task_id: int, duration: float = 0.1) -> dict:
            """Task that can be executed in parallel."""
            time.sleep(duration)
            return {"task_id": task_id, "completed": True, "duration": duration}

        toolkit.add(parallel_task)

        task_configs = [{"task_id": i, "duration": 0.05} for i in range(10)]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(toolkit.run_tool, "parallel_task", config)
                for config in task_configs
            ]

            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        end_time = time.time()
        total_time = end_time - start_time

        sequential_time = len(task_configs) * 0.05
        assert total_time < sequential_time * 0.8, (
            "Parallel execution not significantly faster"
        )

        assert len(results) == 10
        completed_task_ids = {result["task_id"] for result in results}
        expected_task_ids = {i for i in range(10)}
        assert completed_task_ids == expected_task_ids

    def test_state_management_workflow(self):
        """Test workflow with state management between tools."""
        toolkit = Toolkit()

        workflow_state = {"counter": 0, "history": []}

        @tool_call
        def state_updater(action: str, value: int) -> dict:
            """Tool that updates workflow state."""
            if action == "increment":
                workflow_state["counter"] += value
            elif action == "decrement":
                workflow_state["counter"] -= value
            elif action == "reset":
                workflow_state["counter"] = 0

            workflow_state["history"].append(
                {
                    "action": action,
                    "value": value,
                    "counter_after": workflow_state["counter"],
                }
            )

            return {
                "action": action,
                "current_counter": workflow_state["counter"],
                "history_length": len(workflow_state["history"]),
            }

        @tool_call
        def state_reader() -> dict:
            """Tool that reads workflow state."""
            return {
                "current_counter": workflow_state["counter"],
                "total_operations": len(workflow_state["history"]),
                "last_operation": workflow_state["history"][-1]
                if workflow_state["history"]
                else None,
            }

        toolkit.add_tools(state_updater, state_reader)

        operations = [
            {"action": "increment", "value": 5},
            {"action": "increment", "value": 3},
            {"action": "decrement", "value": 2},
            {"action": "increment", "value": 1},
        ]

        for operation in operations:
            result = toolkit.run_tool("state_updater", operation)
            assert result["action"] == operation["action"]

        final_state = toolkit.run_tool("state_reader", {})

        assert final_state["current_counter"] == 7
        assert final_state["total_operations"] == 4
        assert final_state["last_operation"]["action"] == "increment"
        assert final_state["last_operation"]["value"] == 1

    def test_markdown_generation_integration(self):
        """Test integration with markdown generation."""
        toolkit = Toolkit()
        toolkit.add_tools(get_weather, calculate_distance, send_notification)

        markdown_output = toolkit.to_markdown()

        assert "### Tools" in markdown_output
        assert "|" in markdown_output
        assert "get_weather" in markdown_output
        assert "calculate_distance" in markdown_output
        assert "send_notification" in markdown_output

        assert "weather" in markdown_output.lower()
        assert "distance" in markdown_output.lower()
        assert "notification" in markdown_output.lower()

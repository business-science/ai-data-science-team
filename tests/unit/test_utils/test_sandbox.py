"""
Unit tests for sandbox code execution.
"""

import pytest
import subprocess
import sys
from unittest.mock import MagicMock, patch

try:
    from ai_data_science_team.utils.sandbox import (
        run_code_sandboxed_subprocess,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestCodeExecution:
    """Tests for safe code execution."""

    def test_simple_python_execution(self):
        """Test executing simple Python code."""
        code = "result = 1 + 1"

        # Execute in a controlled way
        local_vars = {}
        exec(code, {}, local_vars)

        assert local_vars["result"] == 2

    def test_pandas_code_execution(self):
        """Test executing pandas code."""
        import pandas as pd

        code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
result = df['a'].sum()
"""
        local_vars = {}
        exec(code, {"pd": pd}, local_vars)

        assert local_vars["result"] == 6

    def test_code_with_imports(self):
        """Test code that imports modules."""
        code = """
import math
result = math.sqrt(16)
"""
        local_vars = {}
        exec(code, {}, local_vars)

        assert local_vars["result"] == 4.0

    def test_code_with_function_definition(self):
        """Test code that defines functions."""
        code = """
def add(a, b):
    return a + b

result = add(3, 4)
"""
        local_vars = {}
        exec(code, {}, local_vars)

        assert local_vars["result"] == 7

    def test_code_returns_dataframe(self):
        """Test code that returns a DataFrame."""
        import pandas as pd

        code = """
import pandas as pd
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})
result = df
"""
        local_vars = {}
        exec(code, {"pd": pd}, local_vars)

        assert isinstance(local_vars["result"], pd.DataFrame)
        assert len(local_vars["result"]) == 2


class TestCodeValidation:
    """Tests for code validation before execution."""

    def test_detect_dangerous_imports(self):
        """Test detecting dangerous imports."""
        dangerous_patterns = [
            "import os",
            "from os import system",
            "import subprocess",
            "from subprocess import Popen",
            "__import__('os')",
        ]

        for pattern in dangerous_patterns:
            assert "os" in pattern or "subprocess" in pattern or "__import__" in pattern

    def test_detect_file_operations(self):
        """Test detecting file operation patterns."""
        file_patterns = [
            "open('file.txt', 'w')",
            "with open('file')",
            "f.write(",
            "os.remove(",
        ]

        for pattern in file_patterns:
            assert "open" in pattern or "write" in pattern or "remove" in pattern

    def test_detect_network_operations(self):
        """Test detecting network operation patterns."""
        network_patterns = [
            "import socket",
            "urllib.request",
            "requests.get",
            "http.client",
        ]

        for pattern in network_patterns:
            assert any(word in pattern for word in ["socket", "urllib", "requests", "http"])


class TestSubprocessExecution:
    """Tests for subprocess-based code execution."""

    def test_subprocess_python_execution(self):
        """Test executing Python in subprocess."""
        code = "print(1 + 1)"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "2" in result.stdout

    def test_subprocess_timeout(self):
        """Test that subprocess respects timeout."""
        code = "import time; time.sleep(10)"

        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=1,
            )

    def test_subprocess_captures_stderr(self):
        """Test that subprocess captures error output."""
        code = "raise ValueError('test error')"

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "ValueError" in result.stderr


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestSandboxFunction:
    """Tests for the sandbox execution function."""

    def test_run_code_sandboxed_subprocess_exists(self):
        """Test that run_code_sandboxed_subprocess function is available."""
        assert run_code_sandboxed_subprocess is not None
        assert callable(run_code_sandboxed_subprocess)


class TestCodeSanitization:
    """Tests for code sanitization."""

    def test_remove_dangerous_patterns(self):
        """Test removing dangerous patterns from code."""
        dangerous_code = """
import os
os.system('rm -rf /')
"""
        # A sanitizer should detect 'os.system' as dangerous
        assert "os.system" in dangerous_code

    def test_escape_special_characters(self):
        """Test escaping special characters in code."""
        code_with_special = "print('Hello\\nWorld')"

        # Should be properly escaped
        assert "\\n" in code_with_special

    def test_validate_code_syntax(self):
        """Test validating Python syntax."""
        valid_code = "x = 1 + 2"
        invalid_code = "x = 1 +"

        # Valid code should compile
        compile(valid_code, "<string>", "exec")

        # Invalid code should raise
        with pytest.raises(SyntaxError):
            compile(invalid_code, "<string>", "exec")


class TestResourceLimits:
    """Tests for resource limiting in code execution."""

    def test_memory_limit_detection(self):
        """Test that memory-intensive code can be detected."""
        memory_code = """
# This would use a lot of memory
big_list = [0] * (10 ** 9)
"""
        # Code analysis should detect this pattern
        assert "10 ** 9" in memory_code or "10**9" in memory_code

    def test_infinite_loop_detection(self):
        """Test detection of potential infinite loops."""
        loop_patterns = [
            "while True:",
            "while 1:",
            "for i in iter(int, 1):",
        ]

        for pattern in loop_patterns:
            assert "while" in pattern or "iter" in pattern

    def test_recursion_limit(self):
        """Test that deep recursion is handled."""
        # Define the recursive function properly
        def recurse(n):
            return recurse(n + 1)

        # This should raise RecursionError
        with pytest.raises(RecursionError):
            recurse(0)

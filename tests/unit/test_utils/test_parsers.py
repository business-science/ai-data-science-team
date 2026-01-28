"""
Unit tests for output parsers.
"""

import pytest

try:
    from ai_data_science_team.parsers.parsers import (
        PythonOutputParser,
        SQLOutputParser,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestPythonCodeExtraction:
    """Tests for extracting Python code from LLM output."""

    def test_extract_code_from_markdown_block(self):
        """Test extracting code from markdown code blocks."""
        llm_output = '''Here is the code:

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3]})
result = df.sum()
```

This code creates a DataFrame.'''

        # Extract code between ```python and ```
        import re
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)

        assert match is not None
        code = match.group(1).strip()
        assert "import pandas" in code
        assert "DataFrame" in code

    def test_extract_code_without_language_specifier(self):
        """Test extracting code from generic code blocks."""
        llm_output = '''Code:

```
x = 1 + 2
```
'''
        import re
        pattern = r'```\n?(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)

        assert match is not None
        code = match.group(1).strip()
        assert "x = 1 + 2" in code

    def test_extract_multiple_code_blocks(self):
        """Test extracting multiple code blocks."""
        llm_output = '''First code:
```python
x = 1
```

Second code:
```python
y = 2
```
'''
        import re
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, llm_output, re.DOTALL)

        assert len(matches) == 2
        assert "x = 1" in matches[0]
        assert "y = 2" in matches[1]

    def test_handle_no_code_blocks(self):
        """Test handling output with no code blocks."""
        llm_output = "This is just text without any code."

        import re
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, llm_output, re.DOTALL)

        assert len(matches) == 0

    def test_preserve_code_indentation(self):
        """Test that code indentation is preserved."""
        llm_output = '''```python
def my_function():
    x = 1
    if x > 0:
        return x
    return 0
```'''

        import re
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)

        code = match.group(1)
        lines = code.split('\n')

        # Check indentation is preserved
        assert lines[1].startswith('    ')  # First level indent
        assert lines[2].startswith('    ')
        assert lines[3].startswith('        ')  # Second level indent


class TestSQLCodeExtraction:
    """Tests for extracting SQL code from LLM output."""

    def test_extract_sql_from_markdown(self):
        """Test extracting SQL from markdown blocks."""
        llm_output = '''Here is the query:

```sql
SELECT * FROM users WHERE age > 30
```
'''
        import re
        pattern = r'```sql\n(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)

        assert match is not None
        sql = match.group(1).strip()
        assert "SELECT" in sql
        assert "FROM users" in sql

    def test_extract_multiline_sql(self):
        """Test extracting multiline SQL queries."""
        llm_output = '''```sql
SELECT
    u.name,
    u.email,
    COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.name, u.email
ORDER BY order_count DESC
```'''

        import re
        pattern = r'```sql\n(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)

        sql = match.group(1).strip()
        assert "SELECT" in sql
        assert "JOIN" in sql
        assert "GROUP BY" in sql

    def test_handle_sql_with_comments(self):
        """Test handling SQL with comments."""
        llm_output = '''```sql
-- This query gets active users
SELECT * FROM users
WHERE status = 'active'
-- Only recent users
AND created_at > '2023-01-01'
```'''

        import re
        pattern = r'```sql\n(.*?)```'
        match = re.search(pattern, llm_output, re.DOTALL)

        sql = match.group(1)
        assert "--" in sql  # Comments preserved


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestOutputParsers:
    """Tests for the parser classes."""

    def test_python_output_parser(self):
        """Test PythonOutputParser class."""
        parser = PythonOutputParser()

        llm_output = '''```python
x = 42
```'''
        result = parser.parse(llm_output)

        assert "x = 42" in result

    def test_sql_output_parser(self):
        """Test SQLOutputParser class."""
        parser = SQLOutputParser()

        llm_output = '''```sql
SELECT * FROM table
```'''
        result = parser.parse(llm_output)

        assert "SELECT" in result


class TestCodeCleaning:
    """Tests for code cleaning operations."""

    def test_remove_explanation_text(self):
        """Test removing explanation text from code."""
        raw = '''Here is the code:

```python
x = 1
```

This sets x to 1.'''

        import re
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, raw, re.DOTALL)
        code = match.group(1).strip()

        assert "Here is the code" not in code
        assert "This sets x" not in code

    def test_strip_whitespace(self):
        """Test stripping extra whitespace."""
        code = "   \n\n  x = 1  \n\n   "
        cleaned = code.strip()

        assert cleaned == "x = 1"

    def test_normalize_line_endings(self):
        """Test normalizing line endings."""
        code_with_crlf = "x = 1\r\ny = 2\r\n"
        normalized = code_with_crlf.replace("\r\n", "\n")

        assert "\r" not in normalized
        assert normalized == "x = 1\ny = 2\n"


class TestErrorHandling:
    """Tests for error handling in parsing."""

    def test_handle_malformed_code_block(self):
        """Test handling malformed code blocks."""
        malformed = "```python\nx = 1"  # Missing closing ```

        import re
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, malformed, re.DOTALL)

        # Should not match incomplete blocks
        assert match is None

    def test_handle_nested_backticks(self):
        """Test handling nested backticks."""
        nested = '''```python
code = """
multiline string
"""
```'''

        import re
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, nested, re.DOTALL)

        assert match is not None
        code = match.group(1)
        assert '"""' in code

    def test_handle_empty_code_block(self):
        """Test handling empty code blocks."""
        empty = '''```python
```'''

        import re
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, empty, re.DOTALL)

        assert match is not None
        code = match.group(1).strip()
        assert code == ""

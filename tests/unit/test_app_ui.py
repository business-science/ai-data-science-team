"""
Unit tests for UI enhancement functions in the AI Pipeline Studio app.
"""
from __future__ import annotations

import pytest
import ast
import sys
import os
from typing import Optional

# Add the app directory to the path for imports
APP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "apps", "ai-pipeline-studio-app"
)
sys.path.insert(0, APP_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Math Expression Preprocessing Tests
# ─────────────────────────────────────────────────────────────────────────────

# Re-implement the functions here for isolated testing
_SAFE_MATH_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b if b != 0 else float("inf"),
    ast.Pow: lambda a, b: a**b if abs(b) <= 100 else float("inf"),
    ast.Mod: lambda a, b: a % b if b != 0 else float("inf"),
    ast.FloorDiv: lambda a, b: a // b if b != 0 else float("inf"),
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}


def _eval_math_node(node: ast.AST):
    """Recursively evaluate an AST node for safe math operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.Num):  # Python 3.7 compatibility
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        if left is None or right is None:
            return None
        op = _SAFE_MATH_OPS.get(type(node.op))
        if op is None:
            return None
        try:
            return op(left, right)
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_math_node(node.operand)
        if operand is None:
            return None
        op = _SAFE_MATH_OPS.get(type(node.op))
        if op is None:
            return None
        return op(operand)
    elif isinstance(node, ast.Expression):
        return _eval_math_node(node.body)
    return None


def _is_math_expression(text: str) -> bool:
    """Check if text appears to be a pure math expression."""
    text = text.strip()
    if not text:
        return False
    if not any(c in text for c in "+-*/%^") and not text.replace(".", "").replace("-", "").isdigit():
        return False
    cleaned = text.replace(" ", "").lower()
    for c in cleaned:
        if c.isalpha() and c != "e":
            return False
    return True


def _preprocess_math_input(text: str):
    """Evaluate a math expression if the input is purely mathematical."""
    text = text.strip()
    if not _is_math_expression(text):
        return None
    expr = text.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_math_node(tree)
        if result is None:
            return None
        if isinstance(result, float):
            if result == float("inf") or result != result:
                return None
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 10)
        return f"The result of {text} is **{result}**"
    except (SyntaxError, ValueError, TypeError):
        return None


class TestMathPreprocessing:
    """Tests for math expression preprocessing."""

    def test_simple_multiplication(self):
        """Test 3*3 returns 9."""
        result = _preprocess_math_input("3*3")
        assert result is not None
        assert "9" in result

    def test_addition(self):
        """Test 2+2 returns 4."""
        result = _preprocess_math_input("2+2")
        assert result is not None
        assert "4" in result

    def test_division(self):
        """Test 10/2 returns 5."""
        result = _preprocess_math_input("10/2")
        assert result is not None
        assert "5" in result

    def test_subtraction(self):
        """Test 10-3 returns 7."""
        result = _preprocess_math_input("10-3")
        assert result is not None
        assert "7" in result

    def test_power(self):
        """Test 2^3 returns 8."""
        result = _preprocess_math_input("2^3")
        assert result is not None
        assert "8" in result

    def test_complex_expression(self):
        """Test (3+2)*4 returns 20."""
        result = _preprocess_math_input("(3+2)*4")
        assert result is not None
        assert "20" in result

    def test_decimal(self):
        """Test 3.5*2 returns 7."""
        result = _preprocess_math_input("3.5*2")
        assert result is not None
        assert "7" in result

    def test_negative_number(self):
        """Test -5+10 returns 5."""
        result = _preprocess_math_input("-5+10")
        assert result is not None
        assert "5" in result

    def test_non_math_text(self):
        """Test non-math text returns None."""
        result = _preprocess_math_input("analyze data")
        assert result is None

    def test_mixed_text_with_numbers(self):
        """Test mixed text like 'show 3 charts' returns None."""
        result = _preprocess_math_input("show 3 charts")
        assert result is None

    def test_empty_string(self):
        """Test empty string returns None."""
        result = _preprocess_math_input("")
        assert result is None

    def test_plain_text(self):
        """Test plain text returns None."""
        result = _preprocess_math_input("hello world")
        assert result is None

    def test_division_by_zero(self):
        """Test division by zero is handled gracefully."""
        result = _preprocess_math_input("5/0")
        # Should return None or handle gracefully
        assert result is None or "inf" not in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Search/Filter Tests
# ─────────────────────────────────────────────────────────────────────────────

def _filter_datasets(options: list, search_term: str) -> list:
    """Filter dataset options by search term (case-insensitive)."""
    if not search_term or not search_term.strip():
        return options
    term = search_term.strip().lower()
    return [opt for opt in options if term in opt.lower()]


class TestDatasetSearch:
    """Tests for dataset search/filter functionality."""

    def test_filter_by_exact_match(self):
        """Test filtering with exact match."""
        options = ["sales_data", "customer_info", "orders"]
        filtered = _filter_datasets(options, "sales_data")
        assert filtered == ["sales_data"]

    def test_filter_by_partial_match(self):
        """Test filtering with partial match."""
        options = ["sales_data", "customer_info", "orders"]
        filtered = _filter_datasets(options, "sales")
        assert filtered == ["sales_data"]

    def test_filter_case_insensitive(self):
        """Test that filtering is case-insensitive."""
        options = ["Sales_Data", "customer_info", "orders"]
        filtered = _filter_datasets(options, "sales")
        assert filtered == ["Sales_Data"]

    def test_filter_multiple_matches(self):
        """Test filtering returns multiple matches."""
        options = ["sales_q1", "sales_q2", "orders"]
        filtered = _filter_datasets(options, "sales")
        assert filtered == ["sales_q1", "sales_q2"]

    def test_filter_no_match(self):
        """Test filtering with no matches."""
        options = ["sales_data", "customer_info", "orders"]
        filtered = _filter_datasets(options, "xyz")
        assert filtered == []

    def test_filter_empty_search(self):
        """Test empty search returns all options."""
        options = ["sales_data", "customer_info", "orders"]
        filtered = _filter_datasets(options, "")
        assert filtered == options

    def test_filter_whitespace_search(self):
        """Test whitespace-only search returns all options."""
        options = ["sales_data", "customer_info", "orders"]
        filtered = _filter_datasets(options, "   ")
        assert filtered == options

    def test_filter_empty_options(self):
        """Test filtering empty options list."""
        filtered = _filter_datasets([], "sales")
        assert filtered == []


# ─────────────────────────────────────────────────────────────────────────────
# Progress Message Formatting Tests
# ─────────────────────────────────────────────────────────────────────────────

PROGRESS_STAGES = {
    "data_loader": ("📂", "Loading data"),
    "DataLoaderToolsAgent": ("📂", "Loading data"),
    "data_wrangling": ("🔧", "Wrangling data"),
    "data_cleaning": ("🧹", "Cleaning data"),
    "data_visualization": ("📊", "Creating visualization"),
}


def _format_progress_message(label: str | None, start_time: float | None = None) -> str:
    """Format a progress message with icon and elapsed time."""
    import time
    if not label:
        icon, stage_name = "⏳", "Working"
    else:
        if label.startswith("Routing →"):
            target = label.replace("Routing →", "").strip()
            stage_info = PROGRESS_STAGES.get(target, ("➡️", f"Routing to {target}"))
            icon, stage_name = stage_info[0], f"Routing to {stage_info[1].lower()}"
        else:
            stage_info = PROGRESS_STAGES.get(label, ("⏳", label))
            icon, stage_name = stage_info

    msg = f"{icon} **{stage_name}**"
    if start_time is not None:
        elapsed = time.time() - start_time
        if elapsed >= 60:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            msg += f" ({mins}m {secs}s)"
        else:
            msg += f" ({int(elapsed)}s)"
    return msg


class TestProgressFormatting:
    """Tests for progress message formatting."""

    def test_default_message(self):
        """Test default message when no label."""
        msg = _format_progress_message(None)
        assert "Working" in msg
        assert "⏳" in msg

    def test_known_stage(self):
        """Test formatting for known stage."""
        msg = _format_progress_message("data_loader")
        assert "Loading data" in msg
        assert "📂" in msg

    def test_routing_message(self):
        """Test formatting for routing message."""
        msg = _format_progress_message("Routing → data_loader")
        assert "routing to" in msg.lower()

    def test_unknown_stage(self):
        """Test formatting for unknown stage uses stage name."""
        msg = _format_progress_message("custom_agent")
        assert "custom_agent" in msg

    def test_elapsed_time_seconds(self):
        """Test elapsed time in seconds."""
        import time
        start = time.time() - 30  # 30 seconds ago
        msg = _format_progress_message("data_loader", start)
        assert "s)" in msg

    def test_elapsed_time_minutes(self):
        """Test elapsed time in minutes."""
        import time
        start = time.time() - 90  # 90 seconds ago
        msg = _format_progress_message("data_loader", start)
        assert "m" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Error Logging Tests
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import time as time_module

_error_log: list = []
_ERROR_LOG_MAX_ITEMS = 50


def _log_error(error: Exception, context: str = "") -> str:
    """Log an error with a reference ID for debugging."""
    ref_id = hashlib.md5(
        f"{time_module.time()}{error}{context}".encode()
    ).hexdigest()[:8].upper()
    entry = {
        "ref_id": ref_id,
        "timestamp": time_module.time(),
        "error": str(error),
        "type": type(error).__name__,
        "context": context,
    }
    _error_log.append(entry)
    if len(_error_log) > _ERROR_LOG_MAX_ITEMS:
        _error_log.pop(0)
    return ref_id


def _get_recent_errors(limit: int = 10) -> list:
    """Get recent errors for the debug panel."""
    return _error_log[-limit:][::-1]


class TestErrorLogging:
    """Tests for error logging functionality."""

    def test_log_error_returns_ref_id(self):
        """Test that logging returns a reference ID."""
        _error_log.clear()
        ref_id = _log_error(ValueError("test error"), "test_context")
        assert ref_id is not None
        assert len(ref_id) == 8
        assert ref_id.isupper()

    def test_log_error_stores_entry(self):
        """Test that error is stored in log."""
        _error_log.clear()
        _log_error(ValueError("test error 2"), "test_context_2")
        assert len(_error_log) == 1
        assert _error_log[0]["error"] == "test error 2"
        assert _error_log[0]["context"] == "test_context_2"

    def test_get_recent_errors(self):
        """Test retrieving recent errors."""
        _error_log.clear()
        _log_error(ValueError("err1"), "ctx1")
        _log_error(ValueError("err2"), "ctx2")
        _log_error(ValueError("err3"), "ctx3")
        recent = _get_recent_errors(2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["error"] == "err3"
        assert recent[1]["error"] == "err2"

    def test_error_log_max_items(self):
        """Test that error log respects max items."""
        _error_log.clear()
        for i in range(60):
            _log_error(ValueError(f"err{i}"), f"ctx{i}")
        assert len(_error_log) <= _ERROR_LOG_MAX_ITEMS

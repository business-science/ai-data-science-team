"""
Lightweight sandbox helpers for executing generated code in a separate process.

Goals:
- Keep generated code away from the main process (no in-process exec).
- Block obviously dangerous imports (os, sys, subprocess, socket, etc.).
- Use a minimal builtins set and a network-blocking shim.
- Enforce a timeout from the parent; apply soft resource caps when available.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from textwrap import dedent
from typing import Any, Dict, Tuple


def _build_runner_script() -> str:
    """
    Returns a self-contained Python script (as text) that:
    - Reads JSON from stdin containing: code, function_name, data, data_format, memory_limit_mb.
    - Validates imports against a blocklist.
    - Executes the code with restricted builtins and blocked network.
    - Returns a JSON payload with either {"result": ..., "error": null} or {"result": null, "error": "..."}.
    """

    return dedent(
        r"""
        import ast
        import builtins
        import json
        import sys

        # Optional limits (POSIX-only; no-op on Windows)
        def _apply_resource_limits(memory_limit_mb: int | None):
            try:
                import resource  # type: ignore
            except Exception:
                return
            try:
                if memory_limit_mb:
                    max_bytes = int(memory_limit_mb) * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            except Exception:
                # Failing to set limits should not crash the run; continue without them.
                return

        BLOCKED_MODULES = {
            "os",
            "sys",
            "subprocess",
            "socket",
            "http",
            "urllib",
            "requests",
            "pathlib",
            "shutil",
            "ssl",
            "ftplib",
            "telnetlib",
            "webbrowser",
            "pexpect",
            "psutil",
            "paramiko",
            "ctypes",
        }

        def _reject_blocked_imports(code_text: str):
            tree = ast.parse(code_text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base = (alias.name or "").split(".")[0]
                        if base in BLOCKED_MODULES:
                            raise ImportError(f"Import of '{base}' is blocked.")
                elif isinstance(node, ast.ImportFrom):
                    base = (node.module or "").split(".")[0]
                    if base in BLOCKED_MODULES:
                        raise ImportError(f"Import of '{base}' is blocked.")

        _orig_import = builtins.__import__

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = (name or "").split(".")[0]
            if base in BLOCKED_MODULES:
                raise ImportError(f"Import of '{base}' is blocked.")
            return _orig_import(name, globals, locals, fromlist, level)

        # Minimal builtins; intentionally omit open/exec/eval/compile.
        _SAFE_BUILTINS = {
            name: getattr(builtins, name)
            for name in [
                "abs",
                "all",
                "any",
                "bool",
                "dict",
                "enumerate",
                "float",
                "int",
                "len",
                "list",
                "max",
                "min",
                "range",
                "sum",
                "zip",
                "set",
                "tuple",
                "isinstance",
                "map",
                "filter",
                "sorted",
                "reversed",
                "print",
                "ValueError",
                "TypeError",
                "KeyError",
                "Exception",
                "object",
                "type",
                "getattr",
                "setattr",
                "property",
                "__build_class__",
                "str",
            ]
            if hasattr(builtins, name)
        }
        _SAFE_BUILTINS["__import__"] = _safe_import

        def _block_network():
            try:
                import socket
            except Exception:
                return

            def _blocked(*args, **kwargs):
                raise RuntimeError("Network access is disabled in sandboxed execution.")

            try:
                class NoNetSocket(socket.socket):  # type: ignore
                    def connect(self, *args, **kwargs):
                        raise RuntimeError("Network access is disabled in sandboxed execution.")

                    def connect_ex(self, *args, **kwargs):
                        raise RuntimeError("Network access is disabled in sandboxed execution.")

                socket.socket = NoNetSocket  # type: ignore
            except Exception:
                # Fallback: if subclassing fails, at least block create_connection/getaddrinfo
                pass

            socket.create_connection = _blocked  # type: ignore
            socket.getaddrinfo = _blocked  # type: ignore
            socket.create_server = _blocked  # type: ignore

        def _to_jsonable(obj):
            try:
                import pandas as pd
            except Exception:
                pd = None

            if isinstance(obj, list):
                return [_to_jsonable(item) for item in obj]

            if pd is not None and isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return f"<unserializable: {type(obj).__name__}>"

        def main():
            raw = sys.stdin.buffer.read()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                sys.stdout.write(json.dumps({"result": None, "error": f"Invalid payload: {exc}"}))
                return

            code = payload.get("code") or ""
            function_name = payload.get("function_name")
            data = payload.get("data")
            data_format = payload.get("data_format") or "dataframe"
            memory_limit_mb = payload.get("memory_limit_mb")

            try:
                _reject_blocked_imports(code)
            except Exception as exc:
                sys.stdout.write(json.dumps({"result": None, "error": str(exc)}))
                return

            _apply_resource_limits(memory_limit_mb)
            _block_network()

            try:
                import pandas as pd
                import numpy as np
            except Exception as exc:
                sys.stdout.write(json.dumps({"result": None, "error": f"Sandbox missing dependency: {exc}"}))
                return

            exec_globals = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "__name__": "__main__"}
            local_vars = {}

            try:
                exec(code, exec_globals, local_vars)
            except Exception as exc:
                sys.stdout.write(json.dumps({"result": None, "error": f"Code execution failed: {exc}"}))
                return

            func = local_vars.get(function_name)
            if func is None or not callable(func):
                sys.stdout.write(json.dumps({"result": None, "error": f"Function '{function_name}' not found or not callable."}))
                return

            # Prepare input in the expected format
            try:
                if data_format == "dataframe":
                    input_obj = pd.DataFrame.from_dict(data)
                elif data_format == "dataframe_list":
                    if isinstance(data, list):
                        input_obj = [pd.DataFrame.from_dict(item) for item in data]
                    elif isinstance(data, dict):
                        input_obj = [pd.DataFrame.from_dict(data)]
                    else:
                        raise TypeError(f"Unsupported data type for dataframe_list: {type(data).__name__}")
                else:
                    raise ValueError(f"Unsupported data_format: {data_format}")
            except Exception as exc:
                sys.stdout.write(json.dumps({"result": None, "error": f"Invalid input data: {exc}"}))
                return

            try:
                result = func(input_obj)
                result = _to_jsonable(result)
                sys.stdout.write(json.dumps({"result": result, "error": None}))
            except Exception as exc:
                try:
                    import traceback
                    tb = traceback.format_exc()
                except Exception:
                    tb = ""
                sys.stdout.write(json.dumps({"result": None, "error": f"Function execution failed: {exc}\\n{tb}"}))

        if __name__ == "__main__":
            main()
        """
    ).strip()


SANDBOX_RUNNER_SCRIPT = _build_runner_script()


def run_code_sandboxed_subprocess(
    *,
    code_snippet: str,
    function_name: str,
    data: Any,
    timeout: int = 10,
    memory_limit_mb: int = 512,
    data_format: str = "dataframe",
) -> Tuple[Any, str | None]:
    """
    Execute generated code in a separate Python subprocess with a restricted environment.

    Parameters
    ----------
    code_snippet : str
        The Python code to execute (expected to define `function_name`).
    function_name : str
        The name of the function inside `code_snippet` to invoke.
    data : Any
        Raw data to pass into the function. Shape is interpreted based on `data_format`.
    timeout : int, optional
        Timeout in seconds for the subprocess. Defaults to 10.
    memory_limit_mb : int, optional
        Soft memory cap in MB (best effort; POSIX only). Defaults to 512.
    data_format : str, optional
        Input shape hint for the sandbox. Options:
        - "dataframe": data is a dict and will be converted to a single DataFrame.
        - "dataframe_list": data is a list of dicts (or a single dict) converted to a list of DataFrames.

    Returns
    -------
    tuple
        (result, error_message). `error_message` is None on success.
    """
    payload = {
        "code": code_snippet,
        "function_name": function_name,
        "data": data,
        "memory_limit_mb": memory_limit_mb,
        "data_format": data_format,
    }

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONWARNINGS": "ignore",
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-c", SANDBOX_RUNNER_SCRIPT],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, f"Sandbox timed out after {timeout} seconds."
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Sandbox failed to start: {exc}"

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")

    if completed.returncode != 0:
        err_msg = stderr.strip() or f"Sandbox exited with code {completed.returncode}."
        return None, err_msg

    try:
        parsed = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return None, f"Sandbox returned non-JSON output: {stdout!r}"

    return parsed.get("result"), parsed.get("error")

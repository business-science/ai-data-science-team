# [STATUS: COMPLETED] Sandbox Rollout Plan

## Goal
Replace in-process `exec` across agents with the new subprocess-based sandbox for generated code execution.

## Strategy
- Extend the sandbox helper to support multiple input shapes (single DataFrame and list-of-DataFrames).
- Update each agent’s execute node to call the sandbox with the appropriate `data_format`.
- Keep return values JSON-serializable (DataFrame → dict, list → list of dicts, Plotly dict unchanged).
- Add small tests for success, blocked imports, timeout, and shape handling.

## Agent Order
1) Data Wrangling Agent — supports dict or list-of-dict inputs → pass as list of DataFrames.  
2) Feature Engineering Agent — dict input → DataFrame.  
3) Data Visualization Agent — dict input → DataFrame; return is Plotly dict.  
4) Any shared helpers still using `node_func_execute_agent_code_on_data` to be migrated or retired.

## Open Notes
- Timeouts and memory caps should remain configurable per call; default to 10s / 512MB.
- Consider a global flag to disable network inside the sandbox (already blocked).
- After rollout, deprecate or guard the old in-process executor.

# [STATUS: COMPLETED] Data Cleaning Agent Review

## Overview
Path: `ai_data_science_team/agents/data_cleaning_agent.py`  
Focus: construction of the data cleaning agent (graph wiring, prompts, execution path).  
Result: Several high/medium risk items that need mitigation before wider use.

## Findings (ordered by severity)
- **High – Untrusted code execution (`execute_data_cleaner_code`, ~L665)**  
  The agent `exec`s LLM/user-generated code with full interpreter privileges and real data. There is no sandbox, timeout, or allowlist, so prompt injection can run arbitrary code, exfiltrate data, or damage the host.

- **High – Prompt bloat / token risk (`recommend_cleaning_steps` & `create_data_cleaner_code`, ~L528-535, 561-563)**  
  The prompts inline full `df.head(n_samples)`, `df.describe()`, and `df.info()` for all columns without truncation or column limits. Moderate-size frames will blow past model context limits and materially slow runs.

- **Medium – Bypass path feeds `None` to the LLM (`create_data_cleaner_code`, ~L555-609)**  
  When `bypass_recommended_steps=True`, `{recommended_steps}` is still rendered from state, producing “Recommended Steps: None”. That is ambiguous input and often yields junk code; it should fall back to the default cleaning steps.

- **Medium – Wrapper methods drop results & skip BaseAgent cleanup (`invoke_agent`/`ainvoke_agent`, ~L192-270)**  
  These methods call the compiled graph directly, return `None`, and bypass `BaseAgent.invoke/ainvoke` logic (e.g., duplicate-message cleanup). Callers expecting a response get `None`, and message normalization is skipped.

- **Low – Log path default tied to `os.getcwd()` (`LOG_PATH` and setup, ~L43-45, 448-453)**  
  Import-time `LOG_PATH` may write to unexpected or unwritable locations when used as a library. This can surprise consumers or fail silently.

## Recommended Fixes
1) **Contain code execution**
   - Run generated code in an isolated process with a minimal globals/builtins map and a time/memory budget (e.g., `multiprocessing` or a small subprocess that only receives the code and data).  
   - At minimum, restrict `exec` globals (drop `__builtins__` or provide a safe allowlist), and consider `ast` checks for dangerous imports/calls before execution.

2) **Constrain prompt payloads**
   - Limit columns/rows included in summaries (e.g., cap columns, sample rows, shorten `info()`, optionally skip `describe()` for wide frames).  
   - Add truncation (character/line caps) and make `n_samples` guard both rows and columns. Consider a `skip_stats`/`max_columns` toggle exposed to callers.

3) **Fix bypass fallback**
   - If `bypass_recommended_steps` is True and `recommended_steps` is missing, inject the documented default cleaning steps instead of `None` before calling the LLM.

4) **Return responses & reuse BaseAgent hooks**
   - Have `invoke_agent`/`ainvoke_agent` delegate to `self.invoke`/`self.ainvoke` (or return the compiled graph response) so callers receive the result and message deduping runs.

5) **Safer log path default**
   - Default `log_path` to a project-relative path or require an explicit path when `log=True`. Consider falling back to a temp directory if missing.

## Next Steps
- Confirm the desired containment approach for executing generated code (subprocess vs. restricted globals).  
- Implement fixes in priority order: (1) execution safety, (2) prompt truncation, (3) bypass fallback, (4) wrapper return behavior, (5) log path default.  
- Add targeted tests: prompt truncation for wide frames, bypass fallback behavior, invoke/ainvoke return value expectations, and a smoke test for the sandboxed execution path.

## LangChain Conversation Migration Plan

Goal: migrate agents to a message-first structure (`HumanMessage`/`AIMessage`) for compatibility with supervisor/team architectures, without breaking existing APIs.

### Principles
- Keep backward compatibility: preserve `invoke_agent(user_instructions=...)`; add message-based entry.
- Single source of truth: state carries `messages: Sequence[BaseMessage]`.
- Tool transparency: include tool-call messages (or summaries) when desired.

### Steps (apply per agent)
1) **Inputs**: accept `messages` (preferred) or wrap `user_instructions` into `[HumanMessage(content=...)]`. Normalize into state as `messages`.
2) **State schema**: include `messages: Annotated[Sequence[BaseMessage], operator.add]` plus agent-specific fields (data, errors, summaries).
3) **Outputs**: append an `AIMessage` with the agent’s answer into `messages`. Keep existing result keys (e.g., `data_wrangled`, `recommended_steps`) for compatibility.
4) **Tool calls**: for ReAct/tool agents, optionally include tool call messages in `messages`. Add a flag (e.g., `log_tool_calls=True`) to control transcript verbosity.
5) **Accessors**: update getters to read from `messages` (last assistant message). Keep helpers to return artifacts/data separately.
6) **Entry nodes**: first node ensures `messages` exists (wrap string input) before downstream logic.
7) **Supervisors/teams**: with `messages` standardized, supervisors can route and evaluate full conversations.

### Rollout order
1) Pilot on one agent (e.g., `data_loader_tools_agent`).
2) Extend to utility agents (cleaning, wrangling, visualization).
3) Extend to SQL and feature engineering agents.
4) Update team/supervisor harnesses to rely solely on `messages`.

### Risks & Mitigations
- **Transcript bloat**: use `log_tool_calls` to toggle tool-call messages.
- **Existing callers expect strings**: keep `invoke_agent` signature and string-returning getters.
- **State drift**: always return fresh `messages` from nodes; avoid mutating external state.

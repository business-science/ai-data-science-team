"""Deterministic graph + MCP SDK tests; the model and HTTP server are scripted."""

import asyncio
import json
import re
import unittest
from unittest.mock import patch
from uuid import UUID

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from ai_data_science_team import ResearchToolsAgent
from ai_data_science_team.tools.web_research import MCP_URL, USER_AGENT, ParallelSearchClient

SOURCE_URL = "https://pandas.pydata.org/docs/user_guide/io.html"


class ScriptedResearchModel(BaseChatModel):
    """Request search then fetch, and consume the actual resulting tool text."""

    _tools: set = PrivateAttr(default_factory=set)
    _schemas: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-research-test"

    def bind_tools(self, tools, **kwargs):
        self._tools = {t.name for t in tools}
        self._schemas = [t.tool_call_schema.model_json_schema() for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_user = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        request = messages[last_user].content
        outputs = [m for m in messages[last_user + 1:] if isinstance(m, ToolMessage)]
        if not self._tools:
            content, calls = "No network tools available.", []
        elif not outputs:
            if "custom_search" in self._tools:
                name, args = "custom_search", {"query": request}
            elif request.startswith("fetch "):
                name, args = "web_fetch", {"urls": [request[6:]]}
            else:
                name, args = "web_search", {
                    "objective": "Find the official pandas CSV loading documentation.",
                    "search_queries": ["pandas read_csv official documentation"],
                }
            content, calls = "", [{"name": name, "args": args, "id": f"call-{len(messages)}"}]
        elif request == "research" and len(outputs) == 1 and outputs[0].status != "error":
            # Read the source URL from the search output instead of a canned answer.
            found = re.search(r'https://[^\s"<>]+', str(outputs[0].content))
            content, calls = "", [{"name": "web_fetch", "args": {"urls": [found[0]]},
                                    "id": f"call-{len(messages)}"}]
        else:
            content, calls = "Evidence: " + "\n".join(str(m.content) for m in outputs), []
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=calls))])


class MCPServer:
    def __init__(self):
        self.requests = []
        self.calls = []
        self.fault = None
        self.started = asyncio.Event()
        self.cancelled = False

    async def handle(self, request):
        data = json.loads(await request.aread()) if request.method == "POST" else {}
        self.requests.append((str(request.url), dict(request.headers), data))
        method = data.get("method")
        if method == "initialize":
            result = {"protocolVersion": data["params"]["protocolVersion"],
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                 for name in ("web_search", "web_fetch")]}
        elif method == "tools/call":
            self.calls.append(data["params"])
            self.started.set()
            if self.fault == "slow":
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True
            if self.fault == "disconnect":
                raise httpx.ConnectError("fixture disconnect", request=request)
            error = self.fault == "quota"
            text = "Anonymous quota reached; try later." if error else (
                f"Dataset documentation {SOURCE_URL}" if data["params"]["name"] == "web_search"
                else f"Source {SOURCE_URL}: read_csv supports missing-value options."
            )
            result = {"content": [{"type": "text", "text": text}], "isError": error}
        elif "id" not in data:
            return httpx.Response(202)
        else:
            raise AssertionError(f"Unexpected method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": data["id"], "result": result})

    def patch(self):
        original = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(self.handle)
            return original(*args, **kwargs)

        return patch("httpx.AsyncClient", side_effect=factory)


def agent(**kwargs):
    return ResearchToolsAgent(model=ScriptedResearchModel(), **kwargs)


def config(thread):
    return {"configurable": {"thread_id": thread}}


class ResearchTests(unittest.TestCase):
    def test_anonymous_default_dispatch_and_consumption(self):
        server = MCPServer()
        with server.patch(), patch.dict("os.environ", {"PARALLEL_API_KEY": "ambient-must-not-be-used"}):
            a = agent()
            a.invoke_agent("research")
        self.assertIn("read_csv supports missing-value options", a.get_ai_message())
        self.assertIn(SOURCE_URL, a.get_ai_message())
        self.assertEqual([c["name"] for c in server.calls], ["web_search", "web_fetch"])
        ids = {c["arguments"]["session_id"] for c in server.calls}
        self.assertEqual(len(ids), 1)
        UUID(next(iter(ids)))
        self.assertEqual({r[0] for r in server.requests}, {MCP_URL})
        self.assertTrue({"initialize", "tools/list", "tools/call"}.issubset({r[2].get("method") for r in server.requests}))
        for _, headers, _ in server.requests:
            self.assertEqual(headers["user-agent"], USER_AGENT)
            self.assertNotIn("authorization", headers)
        for schema in a._params["model"]._schemas:
            self.assertNotIn("session_id", schema["properties"])

    def test_threads_followups_quota_retry_and_reset(self):
        server = MCPServer()
        shared_client = ParallelSearchClient()
        a = agent(client=shared_client, checkpointer=InMemorySaver())
        with server.patch():
            a.invoke_agent("research", config=config("a"))
            a.invoke_agent("search again", config=config("b"))
            server.fault = "quota"
            a.invoke_agent("search again", config=config("a"))
            self.assertIn("quota", a.get_ai_message())
            server.fault = None
            a.invoke_agent("search again", config=config("a"))
            a.invoke_agent("search again", config=config("new-chat"))
        ids = [c["arguments"]["session_id"] for c in server.calls]
        self.assertEqual(ids[0], ids[1])  # search then fetch
        self.assertEqual(ids[0], ids[3])  # follow-up quota
        self.assertEqual(ids[0], ids[4])  # retry
        self.assertEqual(len({ids[0], ids[2], ids[5]}), 3)

    def test_explicit_tools_replace_default_and_disable_wins(self):
        @tool
        def custom_search(query: str) -> str:
            """Search an explicitly configured source."""
            return "custom source: " + query

        server = MCPServer()
        with server.patch():
            a = agent(tools=[custom_search])
            a.invoke_agent("search")
            self.assertIn("custom source", a.get_ai_message())
            for opts in ({"tools": []}, {"search_enabled": False},
                         {"tools": [custom_search], "search_enabled": False}):
                disabled = agent(**opts)
                disabled.invoke_agent("search")
                self.assertIn("No network tools", disabled.get_ai_message())
        self.assertEqual(server.requests, [])

    def test_saved_conversation_survives_agent_reconstruction(self):
        server = MCPServer()
        saver = InMemorySaver()
        with server.patch():
            first = agent(checkpointer=saver)
            first.invoke_agent("search", config=config("saved"))
            restored = agent(checkpointer=saver)
            restored.invoke_agent("search again", config=config("saved"))
        self.assertEqual(server.calls[0]["arguments"]["session_id"],
                         server.calls[1]["arguments"]["session_id"])

    def test_argument_bounds_and_invalid_timeouts(self):
        server = MCPServer()
        with server.patch():
            # Exercise invalid model output through the graph's ToolNode.
            class InvalidArgumentsModel(ScriptedResearchModel):
                def _generate(self, messages, **kwargs):
                    if isinstance(messages[-1], ToolMessage):
                        return ChatResult(generations=[ChatGeneration(message=AIMessage(
                            content=str(messages[-1].content)))])
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[
                        {"name": "web_fetch", "args": {"urls": []}, "id": "invalid"}
                    ]))])

            a = ResearchToolsAgent(model=InvalidArgumentsModel())
            a.invoke_agent("fetch")
            self.assertIn("validation error", a.get_ai_message().lower())
        self.assertEqual(server.requests, [])
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                ParallelSearchClient(timeout=timeout)

    def test_approval_pause_has_no_network_effect(self):
        server = MCPServer()
        a = agent(checkpointer=InMemorySaver(), create_react_agent_kwargs={"interrupt_before": ["tools"]})
        with server.patch():
            a.invoke_agent("search", config=config("approval"))
            self.assertEqual(server.requests, [])
            a.invoke(None, config=config("approval"))
        self.assertEqual(len(server.calls), 1)
        self.assertIn(SOURCE_URL, a.get_ai_message())

    def test_invalid_urls_never_reach_mcp(self):
        server = MCPServer()
        with server.patch():
            for url in ("file:///etc/passwd", "https://name:secret@example.org/", "javascript:alert(1)",
                        "https://example.org:bad", "http://"):
                a = agent()
                a.invoke_agent("fetch " + url)
                self.assertIn("HTTP(S)", a.get_ai_message())
        self.assertEqual(server.requests, [])

    def test_explicit_headers_preserved(self):
        server = MCPServer()
        with server.patch():
            a = agent(client=ParallelSearchClient(headers={"Authorization": "Bearer explicit-test-key",
                                                          "User-Agent": "wrong"}))
            a.invoke_agent("search")
        for _, headers, _ in server.requests:
            self.assertEqual(headers["authorization"], "Bearer explicit-test-key")
            self.assertEqual(headers["user-agent"], USER_AGENT)

    def test_transport_error_is_visible_without_fallback(self):
        server = MCPServer()
        server.fault = "disconnect"
        with server.patch():
            a = agent()
            a.invoke_agent("research")
        self.assertIn("request failed", a.get_ai_message())
        self.assertEqual(len(server.calls), 1)


class AsyncResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_threads_shared_client_and_separate_agents(self):
        server = MCPServer()
        shared_client = ParallelSearchClient()
        a = agent(client=shared_client, checkpointer=InMemorySaver())
        b = agent(client=shared_client)
        with server.patch():
            outputs = await asyncio.gather(
                a.ainvoke({"messages": [("user", "research")]}, config=config("one")),
                a.ainvoke({"messages": [("user", "research")]}, config=config("two")),
                b.ainvoke({"messages": [("user", "research")]}),
            )
            await a.ainvoke_agent("search again", config=config("one"))
        ids = [o["search_session_id"] for o in outputs]
        self.assertEqual(len(set(ids)), 3)
        for session_id in ids:
            calls = [c for c in server.calls if c["arguments"]["session_id"] == session_id]
            self.assertEqual([c["name"] for c in calls[:2]], ["web_search", "web_fetch"])
        self.assertEqual(server.calls[-1]["arguments"]["session_id"], ids[0])

    async def test_total_timeout_and_cancellation(self):
        for cancel in (False, True):
            server = MCPServer()
            server.fault = "slow"
            a = agent(client=ParallelSearchClient(timeout=30 if cancel else 0.1))
            with server.patch():
                task = asyncio.create_task(a.ainvoke_agent("search"))
                await asyncio.wait_for(server.started.wait(), 5)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    await asyncio.wait_for(task, 5)
                    self.assertIn("timed out", a.get_ai_message())
            self.assertTrue(server.cancelled)
            self.assertEqual(len(server.calls), 1)


if __name__ == "__main__":
    unittest.main()

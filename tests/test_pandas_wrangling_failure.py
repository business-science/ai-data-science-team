"""Offline regression tests for analyst error propagation and routing."""

import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from ai_data_science_team.multiagents.pandas_data_analyst import (
    make_pandas_data_analyst,
)


class WranglingFailureTests(unittest.TestCase):
    def make_graph(self, route, error=None, checkpointer=None):
        model = RunnableLambda(
            lambda _: '{"routing_preprocessor_decision": "' + route + '"}'
        )
        self.wrangler = Mock()
        self.wrangler.invoke.return_value = {
            "messages": [AIMessage(content="Wrangler report")],
            "data_wrangled": None if error else {"value": {0: 2}},
            "data_wrangler_error": error,
        }
        self.viz = Mock()
        self.viz.invoke.return_value = {
            "messages": [],
            "plotly_graph": {"data": []},
            "data_visualization_error": None,
        }
        return make_pandas_data_analyst(model, self.wrangler, self.viz, checkpointer)

    def invoke(self, graph, **kwargs):
        return graph.invoke(
            {
                "user_instructions": "Analyze",
                "data_raw": {"value": {0: 1}},
                "max_retries": 3,
                "retry_count": 0,
            },
            **kwargs
        )

    def test_failure_is_reported_on_both_routes(self):
        for route in ("table", "chart"):
            with self.subTest(route=route):
                result = self.invoke(self.make_graph(route, "Transformation failed"))
                self.assertEqual(
                    result.get("data_wrangler_error"), "Transformation failed"
                )
                self.assertIn("Transformation failed", result["messages"][-1].content)
                self.assertNotIn("Workflow completed", result["messages"][-1].content)
                self.viz.invoke.assert_not_called()
                self.assertIsNone(result.get("plotly_graph"))
                self.assertTrue(
                    any(
                        message.content == "Wrangler report"
                        for message in result["messages"]
                    )
                )

    def test_successful_chart_uses_wrangled_data(self):
        result = self.invoke(self.make_graph("chart"))
        self.assertEqual(
            self.viz.invoke.call_args.args[0]["data_raw"], {"value": {0: 2}}
        )
        self.assertIn("Chart created", result["messages"][-1].content)

    def test_successful_table_skips_visualization(self):
        result = self.invoke(self.make_graph("table"))
        self.viz.invoke.assert_not_called()
        self.assertIn("1 rows x 1 cols", result["messages"][-1].content)

    def test_failed_turn_does_not_reuse_previous_chart(self):
        graph = self.make_graph("chart", checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "regression"}}
        self.invoke(graph, config=config)
        self.viz.invoke.reset_mock()
        self.wrangler.invoke.return_value = {
            "messages": [],
            "data_wrangled": None,
            "data_wrangler_error": "Failed again",
        }
        result = self.invoke(graph, config=config)
        self.assertIsNone(result.get("plotly_graph"))
        self.assertIn("Failed again", result["messages"][-1].content)
        self.viz.invoke.assert_not_called()

    def test_successful_followup_clears_previous_error(self):
        graph = self.make_graph("chart", "Failed", InMemorySaver())
        config = {"configurable": {"thread_id": "recovery"}}
        self.invoke(graph, config=config)
        self.wrangler.invoke.return_value = {
            "messages": [],
            "data_wrangled": {"value": {0: 2}},
            "data_wrangler_error": None,
        }
        result = self.invoke(graph, config=config)
        self.assertIsNone(result.get("data_wrangler_error"))
        self.viz.invoke.assert_called_once()
        self.assertIn("Chart created", result["messages"][-1].content)

    def test_visualization_failure_is_still_reported(self):
        graph = self.make_graph("chart")
        self.viz.invoke.return_value = {
            "messages": [],
            "plotly_graph": None,
            "data_visualization_error": "Plot failed",
        }
        result = self.invoke(graph)
        self.assertIn("Chart not created: Plot failed", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()

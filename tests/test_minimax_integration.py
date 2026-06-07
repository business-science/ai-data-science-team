"""Integration tests for MiniMax provider.

These tests make real API calls to the MiniMax API.
They require the MINIMAX_API_KEY environment variable to be set.
Skipped automatically if the key is not available.
"""

import os
import unittest

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxLiveAPI(unittest.TestCase):
    """Integration tests that call the MiniMax API."""

    def _make_llm(self, model: str = "MiniMax-M3", **kwargs):
        return ChatOpenAI(
            model=model,
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
            temperature=0.7,
            **kwargs,
        )

    def test_simple_invoke(self):
        """MiniMax M3 should respond to a simple prompt."""
        llm = self._make_llm()
        response = llm.invoke([HumanMessage(content="Say hello in one word.")])
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    def test_m27_model(self):
        """MiniMax M2.7 should also work."""
        llm = self._make_llm(model="MiniMax-M2.7")
        response = llm.invoke([HumanMessage(content="What is 2+2? Answer with just the number.")])
        self.assertIsNotNone(response)
        self.assertIn("4", response.content)

    def test_data_science_prompt(self):
        """MiniMax should handle data-science-style prompts (relevant to this project)."""
        llm = self._make_llm()
        prompt = (
            "Write a Python function called `clean_missing` that takes a pandas DataFrame "
            "and removes columns with more than 40% missing values. Return only the function code."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        self.assertIsNotNone(response)
        self.assertIn("def", response.content)
        self.assertIn("clean_missing", response.content)


if __name__ == "__main__":
    unittest.main()

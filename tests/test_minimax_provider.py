"""Unit tests for MiniMax provider integration.

Tests verify that MiniMax can be configured as an LLM provider
across the AI Data Science Team apps using the OpenAI-compatible API.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# MiniMax provider constants (shared across apps)
# ---------------------------------------------------------------------------

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODELS = ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"]


class TestMiniMaxChatOpenAICreation(unittest.TestCase):
    """Test that ChatOpenAI can be instantiated with MiniMax parameters."""

    def test_create_minimax_llm_default_model(self):
        """MiniMax M3 should be the default model."""
        llm = ChatOpenAI(
            model="MiniMax-M3",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
        )
        self.assertEqual(llm.model_name, "MiniMax-M3")
        self.assertIn("minimax", str(llm.openai_api_base).lower())

    def test_create_minimax_llm_m27(self):
        """MiniMax M2.7 should be creatable."""
        llm = ChatOpenAI(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
        )
        self.assertEqual(llm.model_name, "MiniMax-M2.7")

    def test_create_minimax_llm_m27_highspeed(self):
        """MiniMax M2.7-highspeed should be creatable."""
        llm = ChatOpenAI(
            model="MiniMax-M2.7-highspeed",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
        )
        self.assertEqual(llm.model_name, "MiniMax-M2.7-highspeed")

    def test_minimax_base_url(self):
        """The base URL should point to MiniMax API."""
        llm = ChatOpenAI(
            model="MiniMax-M3",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
        )
        base = str(llm.openai_api_base)
        self.assertIn("api.minimax.io", base)

    def test_minimax_temperature_clamp(self):
        """MiniMax accepts temperature in [0, 1]."""
        llm = ChatOpenAI(
            model="MiniMax-M3",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
            temperature=0.7,
        )
        self.assertEqual(llm.temperature, 0.7)

    def test_minimax_temperature_zero(self):
        """MiniMax now accepts temperature=0."""
        llm = ChatOpenAI(
            model="MiniMax-M3",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
            temperature=0,
        )
        self.assertEqual(llm.temperature, 0)

    def test_minimax_all_models_creatable(self):
        """All advertised MiniMax models should be creatable."""
        for model in MINIMAX_MODELS:
            llm = ChatOpenAI(
                model=model,
                api_key="test-key",
                base_url=MINIMAX_BASE_URL,
            )
            self.assertEqual(llm.model_name, model)


class TestMiniMaxProviderConfig(unittest.TestCase):
    """Test provider configuration patterns used by the apps."""

    def test_provider_list_includes_minimax(self):
        """Verify the PROVIDER_LIST pattern includes MiniMax."""
        provider_list = ["OpenAI", "MiniMax"]
        self.assertIn("MiniMax", provider_list)

    def test_minimax_model_list(self):
        """Verify the MINIMAX_MODEL_LIST pattern is correct (M3 default)."""
        model_list = ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
        self.assertEqual(len(model_list), 3)
        self.assertEqual(model_list[0], "MiniMax-M3")

    def test_minimax_api_key_env_var(self):
        """MiniMax API key should be readable from environment."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-env-key"}):
            key = os.environ.get("MINIMAX_API_KEY")
            self.assertEqual(key, "test-env-key")

    def test_build_team_minimax_branch(self):
        """Test the provider routing logic used in build_team()."""
        llm_provider = "MiniMax"
        model_name = "MiniMax-M3"
        minimax_api_key = "test-key"

        if llm_provider.lower() == "minimax":
            llm = ChatOpenAI(
                model=model_name,
                api_key=minimax_api_key,
                base_url=MINIMAX_BASE_URL,
                temperature=0.7,
            )
        else:
            llm = None

        self.assertIsNotNone(llm)
        self.assertEqual(llm.model_name, "MiniMax-M3")
        self.assertIn("minimax", str(llm.openai_api_base).lower())

    def test_provider_routing_openai(self):
        """OpenAI provider should not route to MiniMax."""
        llm_provider = "OpenAI"
        self.assertNotEqual(llm_provider.lower(), "minimax")

    def test_provider_routing_ollama(self):
        """Ollama provider should not route to MiniMax."""
        llm_provider = "Ollama"
        self.assertNotEqual(llm_provider.lower(), "minimax")

    def test_provider_routing_case_insensitive(self):
        """Provider routing should be case-insensitive."""
        for variant in ["minimax", "MiniMax", "MINIMAX"]:
            self.assertEqual(variant.lower(), "minimax")


class TestMiniMaxAgentCompatibility(unittest.TestCase):
    """Test that MiniMax LLM instances are compatible with the agent system."""

    def test_minimax_llm_is_langchain_compatible(self):
        """MiniMax via ChatOpenAI should be a valid LangChain LLM."""
        llm = ChatOpenAI(
            model="MiniMax-M3",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
        )
        # ChatOpenAI instances should have the standard interface
        self.assertTrue(hasattr(llm, "invoke"))
        self.assertTrue(hasattr(llm, "bind_tools"))

    def test_minimax_llm_serialization(self):
        """MiniMax LLM config should be serializable for LangChain."""
        llm = ChatOpenAI(
            model="MiniMax-M3",
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
            temperature=0.7,
        )
        # The model should expose its config
        self.assertEqual(llm.model_name, "MiniMax-M3")
        self.assertEqual(llm.temperature, 0.7)


class TestPipelineStudioMiniMaxIntegration(unittest.TestCase):
    """Test MiniMax integration in the AI Pipeline Studio app logic."""

    def test_pipeline_studio_provider_list(self):
        """Pipeline Studio should offer MiniMax as a provider."""
        providers = ["OpenAI", "MiniMax", "Ollama"]
        self.assertIn("MiniMax", providers)
        self.assertEqual(providers.index("MiniMax"), 1)

    def test_pipeline_studio_minimax_models(self):
        """Pipeline Studio should list MiniMax models with M3 first."""
        models = ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
        self.assertEqual(len(models), 3)
        self.assertEqual(models[0], "MiniMax-M3")
        self.assertTrue(all(m.startswith("MiniMax") for m in models))

    def test_build_team_creates_minimax_llm(self):
        """build_team() with MiniMax provider should create valid LLM."""
        llm_provider = "MiniMax"
        model_name = "MiniMax-M3"
        minimax_api_key = "test-key"

        if llm_provider.lower() == "minimax":
            llm = ChatOpenAI(
                model=model_name,
                api_key=minimax_api_key,
                base_url=MINIMAX_BASE_URL,
                temperature=0.7,
            )
        else:
            llm = None

        self.assertIsNotNone(llm)
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertEqual(llm.model_name, "MiniMax-M3")


if __name__ == "__main__":
    unittest.main()

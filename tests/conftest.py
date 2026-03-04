"""Shared fixtures for SAGA test suite."""
import pytest
from saga.config import SagaConfig, ApiKeysConfig, ModelsConfig, PromptCachingConfig
from saga.llm.client import LLMClient


@pytest.fixture
def mock_config():
    """Minimal SagaConfig with dummy API keys for unit tests."""
    return SagaConfig(
        api_keys=ApiKeysConfig(
            anthropic="sk-ant-test-key",
            openai="sk-openai-test-key",
            google="google-test-key",
        ),
        models=ModelsConfig(
            narration="claude-sonnet-4-5-20250929",
            extraction="gemini-2.0-flash",
        ),
        prompt_caching=PromptCachingConfig(enabled=True),
    )


@pytest.fixture
def mock_config_non_claude():
    """Config with non-Claude narration model."""
    return SagaConfig(
        api_keys=ApiKeysConfig(
            anthropic="sk-ant-test-key",
            openai="sk-openai-test-key",
            google="google-test-key",
        ),
        models=ModelsConfig(
            narration="gpt-4o",
            extraction="gemini-2.0-flash",
        ),
        prompt_caching=PromptCachingConfig(enabled=True),
    )


@pytest.fixture
def mock_config_caching_disabled():
    """Config with caching explicitly disabled."""
    return SagaConfig(
        api_keys=ApiKeysConfig(
            anthropic="sk-ant-test-key",
            openai="sk-openai-test-key",
            google="google-test-key",
        ),
        models=ModelsConfig(
            narration="claude-sonnet-4-5-20250929",
        ),
        prompt_caching=PromptCachingConfig(enabled=False),
    )


@pytest.fixture
def llm_client(mock_config):
    """LLMClient instance with dummy config."""
    return LLMClient(mock_config)

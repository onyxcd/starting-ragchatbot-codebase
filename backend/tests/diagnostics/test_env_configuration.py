"""Diagnostic tests to verify environment configuration"""

import pytest
import os
from config import Config


def test_api_key_is_set():
    """CRITICAL: Verify ANTHROPIC_API_KEY is configured in environment"""
    config = Config()
    assert config.ANTHROPIC_API_KEY is not None, (
        "ANTHROPIC_API_KEY is None. Check that .env file exists and contains the key."
    )


def test_api_key_not_empty():
    """CRITICAL: Verify ANTHROPIC_API_KEY is not an empty string"""
    config = Config()
    assert config.ANTHROPIC_API_KEY != "", (
        "ANTHROPIC_API_KEY is empty. Add your Anthropic API key to .env file:\n"
        "ANTHROPIC_API_KEY=sk-ant-..."
    )


def test_config_loads_from_env():
    """Verify that Config properly loads from environment variables"""
    # Test with a specific env var
    original_value = os.environ.get("ANTHROPIC_API_KEY")

    # Set test value
    os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

    # Create new config
    config = Config()

    # Verify it loaded
    assert config.ANTHROPIC_API_KEY == "test-key-12345"

    # Restore original value
    if original_value:
        os.environ["ANTHROPIC_API_KEY"] = original_value
    else:
        del os.environ["ANTHROPIC_API_KEY"]


def test_model_name_is_set():
    """Verify that ANTHROPIC_MODEL is configured"""
    config = Config()
    assert config.ANTHROPIC_MODEL is not None
    assert config.ANTHROPIC_MODEL != ""
    assert "claude" in config.ANTHROPIC_MODEL.lower()


def test_chroma_path_is_set():
    """Verify that CHROMA_PATH is configured"""
    config = Config()
    assert config.CHROMA_PATH is not None
    assert config.CHROMA_PATH != ""


def test_embedding_model_is_set():
    """Verify that EMBEDDING_MODEL is configured"""
    config = Config()
    assert config.EMBEDDING_MODEL is not None
    assert config.EMBEDDING_MODEL != ""


def test_config_defaults():
    """Verify that Config has sensible defaults"""
    config = Config()

    # Check chunk settings
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    # Check search settings
    assert config.MAX_RESULTS > 0
    assert config.MAX_HISTORY >= 0

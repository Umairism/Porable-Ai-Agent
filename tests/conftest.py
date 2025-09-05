"""
Test configuration and fixtures for Portable AI Agent
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "ai_engine": {
            "model_path": "test_models/",
            "learning_rate": 0.01,
            "context_window": 128,
            "temperature": 0.5
        },
        "knowledge_base": {
            "embedding_model": "all-MiniLM-L6-v2",
            "max_knowledge_items": 100,
            "similarity_threshold": 0.5
        },
        "memory": {
            "max_conversations": 10,
            "context_length": 5,
            "auto_summarize": False
        }
    }


@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing"""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you tell me about artificial intelligence?"},
        {"role": "assistant", "content": "Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks that typically require human intelligence."}
    ]


@pytest.fixture
def sample_knowledge():
    """Sample knowledge data for testing"""
    return [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "title": "Python Programming",
            "category": "programming",
            "source": "test"
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "title": "Machine Learning",
            "category": "ai",
            "source": "test"
        }
    ]

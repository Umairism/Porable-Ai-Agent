"""
Basic tests for the main application
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all core modules can be imported"""
    try:
        import main
        from core.ai_engine import SelfLearningCore
        from knowledge.knowledge_base import KnowledgeBase
        from memory.conversation_memory import ConversationMemory
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_main_module():
    """Test main module functionality"""
    import main
    
    # Test that main functions exist
    assert hasattr(main, 'main')
    assert hasattr(main, 'print_banner')
    assert hasattr(main, 'check_dependencies')


def test_config_file_exists():
    """Test that config.json exists and is valid"""
    config_path = project_root / "config.json"
    assert config_path.exists(), "config.json file is missing"
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check required sections
    assert "ai_engine" in config
    assert "knowledge_base" in config
    assert "memory" in config


def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = ["core", "knowledge", "memory", "interface", "utils"]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} is missing"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


if __name__ == "__main__":
    test_imports()
    test_main_module()
    test_config_file_exists()
    test_directory_structure()
    print("All basic tests passed!")

#!/usr/bin/env python3
"""
Portable AI Agent - Main Entry Point

A self-contained, offline-capable AI agent with self-learning capabilities.
This is your personal AI assistant that learns from interactions and respects your privacy.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from interface.cli_interface import main as cli_main
from interface.web_interface import main as web_main


def print_banner():
    """Print the application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Portable AI Agent                         ║
║              Offline • Self-Learning • Private               ║
║                                                              ║
║  🤖 Personal AI Assistant                                    ║
║  🧠 Learns from every interaction                            ║
║  🔒 100% offline and private                                 ║
║  📚 Built-in knowledge management                            ║
║  💾 Persistent memory across sessions                        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'torch', 'transformers', 'sentence_transformers', 
        'numpy', 'scipy', 'sklearn', 'faiss', 
        'nltk', 'flask', 'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def setup_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "knowledge", 
        "memory",
        "data",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
    
    print("✅ Directory structure created")


def initialize_agent():
    """Initialize the AI agent components"""
    print("🚀 Initializing Portable AI Agent...")
    
    # Setup directories
    setup_directories()
    
    # Initialize basic knowledge if first run
    knowledge_db = project_root / "knowledge" / "knowledge.db"
    if not knowledge_db.exists():
        print("📚 Setting up initial knowledge base...")
        
        try:
            from knowledge.knowledge_base import KnowledgeBase
            kb = KnowledgeBase()
            
            # Add some basic knowledge
            basic_knowledge = [
                {
                    "content": "I am a portable AI agent that runs completely offline. I learn from our conversations and adapt to your preferences while keeping all data private on your device.",
                    "title": "About me",
                    "category": "system"
                },
                {
                    "content": "You can teach me new things using the 'learn' command, add knowledge with the 'knowledge' command, and see my learning progress with 'stats'.",
                    "title": "How to interact with me",
                    "category": "help"
                },
                {
                    "content": "All conversations, learning, and data processing happen locally. No data is sent to external servers, ensuring complete privacy.",
                    "title": "Privacy and offline operation",
                    "category": "privacy"
                }
            ]
            
            for item in basic_knowledge:
                kb.add_knowledge(
                    content=item["content"],
                    title=item["title"], 
                    category=item["category"],
                    source="initialization"
                )
            
            kb.save_knowledge()
            print("✅ Initial knowledge base created")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize knowledge base: {e}")
    
    print("🎉 Portable AI Agent is ready!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Portable AI Agent - Your personal offline AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start CLI interface
  python main.py --web              # Start web interface
  python main.py --web --port 8080  # Start web interface on port 8080
  python main.py --init             # Initialize/reset the agent
        """
    )
    
    parser.add_argument(
        '--web', 
        action='store_true', 
        help='Start web interface instead of CLI'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Port for web interface (default: 5000)'
    )
    
    parser.add_argument(
        '--host', 
        type=str, 
        default='127.0.0.1',
        help='Host for web interface (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--init', 
        action='store_true',
        help='Initialize/reset the AI agent'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode for web interface'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Initialize if requested or first run
    if args.init or not (project_root / "models").exists():
        initialize_agent()
        if args.init:
            print("✅ Initialization complete!")
            return
    
    # Start the appropriate interface
    try:
        if args.web:
            print(f"🌐 Starting web interface on http://{args.host}:{args.port}")
            print("💡 Open your browser and navigate to the URL above")
            print("🔒 Your AI agent runs locally - no internet required after startup")
            print("Press Ctrl+C to stop the server")
            print("-" * 60)
            
            # Import and start web interface
            from interface.web_interface import WebInterface
            web_interface = WebInterface(host=args.host, port=args.port, debug=args.debug)
            web_interface.run()
            
        else:
            print("💻 Starting command-line interface")
            print("🔒 Your AI agent runs completely offline")
            print("💡 Type 'help' for available commands")
            print("-" * 60)
            
            # Start CLI interface
            cli_main()
            
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using Portable AI Agent!")
        print("🎓 Your AI has learned from this session and will remember next time.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your setup and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

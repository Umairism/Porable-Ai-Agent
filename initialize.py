#!/usr/bin/env python3
"""
Initialization script for Portable AI Agent
Run this first to set up your personal AI assistant
"""

import os
import sys
import subprocess
from pathlib import Path

def print_welcome():
    """Print welcome message"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Portable AI Agent Setup                        ║
║          Setting up your personal AI assistant              ║
╚══════════════════════════════════════════════════════════════╝

🤖 Welcome to your Portable AI Agent setup!

This will prepare your personal AI assistant that:
• Works completely offline
• Learns from your interactions  
• Keeps all data private on your device
• Improves over time through self-learning

Let's get started!
""")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Update pip first
        print("   Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("   Installing required packages...")
        requirements_file = Path(__file__).parent / "requirements.txt"
        
        if requirements_file.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        else:
            # Install essential packages directly
            essential_packages = [
                "torch>=2.0.0",
                "transformers>=4.30.0", 
                "sentence-transformers>=2.2.0",
                "numpy>=1.24.0",
                "scipy>=1.10.0",
                "scikit-learn>=1.3.0",
                "faiss-cpu>=1.7.4",
                "nltk>=3.8",
                "flask>=2.3.0",
                "cryptography>=41.0.0"
            ]
            
            for package in essential_packages:
                print(f"   Installing {package.split('>=')[0]}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n💡 Try manually installing with:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during installation: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directory structure...")
    
    base_path = Path(__file__).parent
    directories = [
        "models",
        "knowledge", 
        "memory",
        "data",
        "logs",
        "interface/templates"
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directory structure created!")

def download_models():
    """Download and setup initial models"""
    print("\n🤖 Setting up AI models...")
    
    try:
        # Import and initialize models
        from sentence_transformers import SentenceTransformer
        
        print("   Downloading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Save model locally
        model_path = Path(__file__).parent / "models" / "sentence_transformer"
        model.save(str(model_path))
        
        print("✅ AI models ready!")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not download models: {e}")
        print("   Models will be downloaded on first use")
        return True  # Not critical for setup

def setup_initial_knowledge():
    """Setup initial knowledge base"""
    print("\n📚 Creating initial knowledge base...")
    
    try:
        from knowledge.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase()
        
        # Add initial knowledge
        initial_knowledge = [
            {
                "content": "I am your Portable AI Agent. I run completely offline on your device and learn from our conversations. I can help you with questions, learn new information, and adapt to your preferences over time.",
                "title": "About your AI assistant",
                "category": "system",
                "importance": 2.0
            },
            {
                "content": "To interact with me, you can: 1) Simply chat normally, 2) Use 'learn' command to teach me specific responses, 3) Use 'knowledge' command to add factual information, 4) Use 'stats' to see my learning progress, 5) Use 'help' for more commands.",
                "title": "How to use your AI assistant",
                "category": "help",
                "importance": 2.0
            },
            {
                "content": "Your privacy is completely protected. All conversations, learning, and data processing happen locally on your device. No information is ever sent to external servers or the internet. You have full control over your data.",
                "title": "Privacy and security",
                "category": "privacy",
                "importance": 2.0
            },
            {
                "content": "I learn continuously through: 1) Analyzing conversation patterns, 2) User feedback and corrections, 3) Self-supervised learning from interactions, 4) Adapting to your communication style and preferences over time.",
                "title": "How self-learning works",
                "category": "learning",
                "importance": 1.5
            },
            {
                "content": "You can access me through: 1) Command-line interface (CLI) - run 'python main.py', 2) Web interface - run 'python main.py --web' and open http://localhost:5000 in your browser.",
                "title": "Interface options",
                "category": "help",
                "importance": 1.5
            }
        ]
        
        for item in initial_knowledge:
            kb.add_knowledge(
                content=item["content"],
                title=item["title"],
                category=item["category"],
                source="initialization",
                importance=item["importance"]
            )
        
        kb.save_knowledge()
        print("✅ Initial knowledge base created!")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not create initial knowledge base: {e}")
        print("   Knowledge base will be created on first use")
        return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\n📝 Creating startup scripts...")
    
    base_path = Path(__file__).parent
    
    # CLI startup script
    cli_script = """#!/bin/bash
# Portable AI Agent - CLI Interface
echo "Starting Portable AI Agent (CLI)..."
python3 main.py
"""
    
    # Web startup script  
    web_script = """#!/bin/bash
# Portable AI Agent - Web Interface
echo "Starting Portable AI Agent (Web Interface)..."
echo "Open http://localhost:5000 in your browser"
python3 main.py --web
"""
    
    # Windows batch files
    cli_batch = """@echo off
echo Starting Portable AI Agent (CLI)...
python main.py
pause
"""
    
    web_batch = """@echo off
echo Starting Portable AI Agent (Web Interface)...
echo Open http://localhost:5000 in your browser
python main.py --web
pause
"""
    
    try:
        # Create Unix scripts
        with open(base_path / "start_cli.sh", "w") as f:
            f.write(cli_script)
        
        with open(base_path / "start_web.sh", "w") as f:
            f.write(web_script)
        
        # Make scripts executable on Unix
        if os.name == 'posix':
            os.chmod(base_path / "start_cli.sh", 0o755)
            os.chmod(base_path / "start_web.sh", 0o755)
        
        # Create Windows batch files
        with open(base_path / "start_cli.bat", "w") as f:
            f.write(cli_batch)
        
        with open(base_path / "start_web.bat", "w") as f:
            f.write(web_batch)
        
        print("✅ Startup scripts created!")
        print("   • start_cli.sh / start_cli.bat - Command-line interface")
        print("   • start_web.sh / start_web.bat - Web interface")
        
    except Exception as e:
        print(f"⚠️  Could not create startup scripts: {e}")

def run_initial_test():
    """Run a basic test to ensure everything works"""
    print("\n🧪 Running initial test...")
    
    try:
        # Test imports
        from core.ai_engine import SelfLearningCore
        from knowledge.knowledge_base import KnowledgeBase
        from memory.conversation_memory import ConversationMemory
        
        print("   Testing AI engine...")
        ai = SelfLearningCore()
        
        print("   Testing knowledge base...")
        kb = KnowledgeBase()
        
        print("   Testing memory system...")
        memory = ConversationMemory()
        
        # Test basic interaction
        print("   Testing basic functionality...")
        response = ai.generate_response("Hello, are you working?", context="test")
        
        if response:
            print("✅ All systems operational!")
            return True
        else:
            print("⚠️  Basic test returned empty response")
            return False
            
    except Exception as e:
        print(f"⚠️  Test failed: {e}")
        print("   The system may still work, but check for errors")
        return False

def print_completion_message():
    """Print setup completion message"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  Setup Complete! 🎉                         ║
╚══════════════════════════════════════════════════════════════╝

🎯 Your Portable AI Agent is ready to use!

🚀 Quick Start:
   • CLI Interface:  python main.py
   • Web Interface:  python main.py --web
   
   Or use the startup scripts:
   • ./start_cli.sh (or start_cli.bat on Windows)
   • ./start_web.sh (or start_web.bat on Windows)

💡 First Steps:
   1. Start chatting normally - your AI learns from every interaction
   2. Use 'help' command to see all available features
   3. Teach specific responses with 'learn' command
   4. Add knowledge with 'knowledge' command
   5. Monitor progress with 'stats' command

🔒 Privacy: Everything runs offline on your device
🧠 Learning: Your AI improves with every conversation
📚 Knowledge: Builds a personal knowledge base over time

Happy chatting with your new AI assistant! 🤖✨
""")

def main():
    """Main setup function"""
    print_welcome()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Setup incomplete due to dependency issues")
        print("You may need to install dependencies manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Download models (optional)
    download_models()
    
    # Setup initial knowledge
    setup_initial_knowledge()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Run test
    run_initial_test()
    
    # Show completion message
    print_completion_message()

if __name__ == "__main__":
    main()

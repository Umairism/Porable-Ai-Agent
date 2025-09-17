#!/usr/bin/env python3
"""
Portable AI Agent - Google Colab Launcher
A simplified launcher for running the Portable AI Agent on Google Colab

Usage:
1. Upload this file to Google Colab
2. Run: !python colab_launcher.py
3. Follow the instructions

Author: Umair Malik
Repository: https://github.com/umairism/Porable-Ai-Agent
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print the application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                🤖 Portable AI Agent                          ║
║                  Google Colab Edition                        ║
║              Offline • Self-Learning • Private               ║
║                                                              ║
║  🚀 Quick Setup for Google Colab                             ║
║  🧠 Learns from every interaction                            ║
║  🔒 Privacy-first AI assistant                               ║
║  📚 Built-in knowledge management                            ║
║  💾 Persistent memory during session                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if we're running in Google Colab"""
    try:
        import google.colab
        print("✅ Running in Google Colab environment")
        return True
    except ImportError:
        print("⚠️ Not running in Google Colab - some features may not work")
        return False

def install_dependencies():
    """Install required dependencies for Colab"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0", 
        "faiss-cpu",
        "flask",
        "nltk",
        "numpy scipy scikit-learn",
        "pandas joblib requests pyyaml",
        "cryptography",
        "pyngrok flask-ngrok",
        "tqdm psutil"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep.split()[0]}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install"] + dep.split(), 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ Warning: Failed to install {dep}")
    
    print("✅ Dependencies installation completed!")

def setup_project():
    """Clone and setup the project"""
    print("📥 Setting up Portable AI Agent...")
    
    # Clone repository if not exists
    if not os.path.exists('/content/Porable-Ai-Agent'):
        print("Cloning repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/umairism/Porable-Ai-Agent.git", 
            "/content/Porable-Ai-Agent"
        ])
    
    # Change to project directory
    os.chdir('/content/Porable-Ai-Agent')
    
    # Add to Python path
    if '/content/Porable-Ai-Agent' not in sys.path:
        sys.path.insert(0, '/content/Porable-Ai-Agent')
    
    print("✅ Project setup completed!")

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK data downloaded!")

def initialize_ai():
    """Initialize AI components"""
    print("🧠 Initializing AI components...")
    
    try:
        # Import AI components
        from core.ai_engine import SelfLearningCore
        from knowledge.knowledge_base import KnowledgeBase
        from memory.conversation_memory import ConversationMemory
        
        print("📚 Initializing Knowledge Base...")
        knowledge_base = KnowledgeBase()
        
        print("💭 Initializing Conversation Memory...")
        conversation_memory = ConversationMemory()
        
        print("🧠 Initializing AI Engine...")
        ai_engine = SelfLearningCore()
        
        # Manually attach components to AI engine
        ai_engine.knowledge_base = knowledge_base
        ai_engine.memory = conversation_memory
        
        # Add a chat method if it doesn't exist
        if not hasattr(ai_engine, 'chat'):
            def chat_method(message):
                try:
                    # Simple response generation
                    response = ai_engine.generate_response(message) if hasattr(ai_engine, 'generate_response') else f"I understand you said: '{message}'. I'm learning from this interaction!"
                    
                    # Store in memory if available
                    if hasattr(ai_engine, 'memory') and ai_engine.memory:
                        ai_engine.memory.add_interaction(message, response)
                    
                    # Update performance metrics
                    if hasattr(ai_engine, 'performance_metrics'):
                        ai_engine.performance_metrics['total_interactions'] += 1
                        ai_engine.performance_metrics['successful_responses'] += 1
                    
                    return response
                except Exception as e:
                    return f"I encountered an issue: {str(e)}, but I'm still learning!"
            
            ai_engine.chat = chat_method
        
        print("✅ AI components initialized!")
        return ai_engine
        
    except Exception as e:
        print(f"⚠️ Using fallback AI: {e}")
        
        class SimpleAI:
            def __init__(self):
                self.memory = []
                self.conversation_count = 0
                
            def chat(self, message):
                self.conversation_count += 1
                response = f"Message {self.conversation_count}: I received '{message}' and I'm learning from it!"
                self.memory.append((message, response))
                return response
        
        return SimpleAI()

def create_web_interface(ai_engine):
    """Create web interface for the AI agent"""
    print("🌐 Creating web interface...")
    
    from flask import Flask, render_template_string, request, jsonify
    
    app = Flask(__name__)
    app.secret_key = 'portable-ai-colab'
    
    # Simple HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Portable AI Agent - Colab</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; background: #fafafa; }
            .message { margin-bottom: 10px; padding: 8px; border-radius: 5px; }
            .user { background: #007bff; color: white; text-align: right; }
            .ai { background: #e9ecef; color: #333; }
            .input-area { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Portable AI Agent</h1>
                <p>Self-Learning AI Assistant - Running on Google Colab</p>
            </div>
            
            <div class="chat-box" id="chatBox">
                <div class="message ai">
                    <strong>🤖 AI:</strong> Hello! I'm your Portable AI Agent. I learn from our conversations. What would you like to talk about?
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Type your message here..." autocomplete="off">
                <button onclick="sendMessage()">Send 🚀</button>
            </div>
        </div>

        <script>
            function addMessage(content, isUser) {
                const chatBox = document.getElementById('chatBox');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;
                messageDiv.innerHTML = `<strong>${isUser ? '👤 You' : '🤖 AI'}:</strong> ${content}`;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;

                addMessage(message, true);
                input.value = '';

                fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => addMessage(data.response, false))
                .catch(error => addMessage('Error: ' + error, false));
            }

            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    '''
    
    @app.route('/')
    def home():
        return render_template_string(html_template)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.get_json()
            message = data.get('message', '')
            
            response = ai_engine.chat(message)
            
            return jsonify({
                'response': response,
                'status': 'success'
            })
        except Exception as e:
            return jsonify({
                'response': f'Error: {str(e)}',
                'status': 'error'
            })
    
    return app

def start_server(app):
    """Start the Flask server with ngrok"""
    print("🚀 Starting server...")
    
    try:
        from pyngrok import ngrok
        import threading
        
        # Try to get ngrok auth token from various sources
        auth_token = None
        
        # Check environment variable
        auth_token = os.environ.get('NGROK_AUTH_TOKEN')
        
        # For Google Colab, try to get from userdata
        if not auth_token:
            try:
                from google.colab import userdata
                auth_token = userdata.get('NGROK_AUTH_TOKEN')
                print("✅ Found ngrok auth token in Colab secrets")
            except:
                pass
        
        # Set auth token if found
        if auth_token:
            ngrok.set_auth_token(auth_token)
            print("🔑 ngrok authentication configured")
        else:
            print("⚠️ No ngrok auth token found. Trying without authentication...")
            print("💡 For better reliability, add NGROK_AUTH_TOKEN to Colab secrets")
        
        # Start Flask in background
        def run_app():
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
        flask_thread = threading.Thread(target=run_app, daemon=True)
        flask_thread.start()
        
        # Wait for Flask to start
        time.sleep(3)
        
        # Create public URL
        public_url = ngrok.connect(5000)
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Your AI Agent is running!")
        print("="*60)
        print(f"🌐 Access your AI at: {public_url}")
        print("\n💡 Features:")
        print("   • Real-time chat interface")
        print("   • Self-learning AI")
        print("   • Conversation memory")
        print("   • Secure tunneled connection")
        print("\n⚠️ Keep this process running to maintain the server!")
        print("="*60)
        
        return public_url
        
    except Exception as e:
        print(f"❌ Error starting server with ngrok: {e}")
        print("\n� Trying alternative local server...")
        
        # Fallback to local server
        try:
            import threading
            
            def run_app():
                app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
            
            flask_thread = threading.Thread(target=run_app, daemon=True)
            flask_thread.start()
            
            time.sleep(2)
            
            print("\n" + "="*60)
            print("🎉 Local server started!")
            print("="*60)
            print("🌐 Access your AI at: http://localhost:5000")
            print("\n💡 Note: This URL only works in Colab environment")
            print("   For public access, add NGROK_AUTH_TOKEN to secrets")
            print("\n⚠️ Keep this process running to maintain the server!")
            print("="*60)
            
            return "http://localhost:5000"
            
        except Exception as e2:
            print(f"❌ Error starting local server: {e2}")
            print("�💬 You can still use the CLI interface")
            return None

def cli_chat(ai_engine):
    """Simple CLI chat interface"""
    print("\n💬 CLI Chat Mode")
    print("="*50)
    print("Type 'quit' to exit")
    
    while True:
        try:
            message = input("\n👤 You: ").strip()
            
            if message.lower() in ['quit', 'exit', 'bye']:
                print("🤖 AI: Goodbye! Thanks for chatting!")
                break
            
            if message:
                response = ai_engine.chat(message)
                print(f"🤖 AI: {response}")
                
        except KeyboardInterrupt:
            print("\n🤖 AI: Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check environment
    is_colab = check_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Setup project
    setup_project()
    
    # Download NLTK data
    download_nltk_data()
    
    # Initialize AI
    ai_engine = initialize_ai()
    
    # Ask user preference
    print("\n🚀 Choose interface:")
    print("1. Web Interface (recommended)")
    print("2. CLI Chat")
    
    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
        
        if choice == "1":
            # Web interface
            app = create_web_interface(ai_engine)
            public_url = start_server(app)
            
            if public_url:
                # Keep server running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Server stopped")
            else:
                # Fallback to CLI
                cli_chat(ai_engine)
        else:
            # CLI interface
            cli_chat(ai_engine)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
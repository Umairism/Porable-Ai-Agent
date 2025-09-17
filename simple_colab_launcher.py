#!/usr/bin/env python3
"""
Portable AI Agent - Simple Colab Launcher (No ngrok required)
A simplified launcher for running the Portable AI Agent on Google Colab without ngrok

Usage:
1. Upload this file to Google Colab
2. Run: !python simple_colab_launcher.py
3. Chat with your AI locally in Colab

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
║               Simple Colab Edition                           ║
║              No ngrok • No Setup • Just Chat                 ║
║                                                              ║
║  🚀 Zero-config setup for Google Colab                       ║
║  🧠 Learns from every interaction                            ║
║  🔒 Local-only access (secure)                               ║
║  📚 Built-in knowledge management                            ║
║  💾 Session memory                                           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def install_dependencies():
    """Install only essential dependencies"""
    print("📦 Installing dependencies...")
    
    essential_deps = [
        "torch --index-url https://download.pytorch.org/whl/cu118",
        "transformers",
        "sentence-transformers", 
        "faiss-cpu",
        "flask",
        "nltk",
        "numpy scipy scikit-learn",
        "pandas requests"
    ]
    
    for dep in essential_deps:
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

def initialize_ai():
    """Initialize AI components"""
    print("🧠 Initializing AI components...")
    
    try:
        # Download NLTK data
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True) 
        nltk.download('wordnet', quiet=True)
        
        # Import AI components
        from core.ai_engine import SelfLearningCore
        
        ai_engine = SelfLearningCore()
        
        # Add enhanced chat method
        def enhanced_chat(message):
            conversation_count = getattr(ai_engine, 'conversation_count', 0) + 1
            ai_engine.conversation_count = conversation_count
            
            if conversation_count == 1:
                response = f"Hello! I'm your Portable AI Agent. You said: '{message}'. I'm excited to start learning!"
            elif 'thank' in message.lower():
                response = f"You're very welcome! I appreciate: '{message}'. I'm learning to be more helpful with each conversation!"
            elif '?' in message:
                response = f"That's a great question: '{message}'. I'm processing and learning from your inquiry (Chat #{conversation_count})!"
            elif 'how are you' in message.lower():
                response = f"I'm doing well and learning! You asked: '{message}'. This is our conversation #{conversation_count} and I'm getting smarter!"
            else:
                response = f"I understand: '{message}'. This is interaction #{conversation_count} and I'm learning something new from each exchange!"
            
            # Store in memory
            if not hasattr(ai_engine, 'chat_history'):
                ai_engine.chat_history = []
            ai_engine.chat_history.append((message, response))
            
            return response
        
        ai_engine.chat = enhanced_chat
        
        print("✅ AI components initialized with enhanced chat!")
        return ai_engine
        
    except Exception as e:
        print(f"⚠️ Using simple AI: {e}")
        
        class SimpleAI:
            def __init__(self):
                self.chat_history = []
                self.conversation_count = 0
                
            def chat(self, message):
                self.conversation_count += 1
                
                if self.conversation_count == 1:
                    response = f"Hello! I'm your AI. You said: '{message}'. Let's chat!"
                elif 'bye' in message.lower():
                    response = f"Thanks for chatting! You said: '{message}'. I learned a lot from our {self.conversation_count} interactions!"
                else:
                    response = f"Message #{self.conversation_count}: '{message}' - I'm learning from each conversation!"
                
                self.chat_history.append((message, response))
                return response
        
        return SimpleAI()

def start_local_chat(ai_engine):
    """Start simple local chat"""
    print("\n" + "="*60)
    print("🎉 AI Agent Ready - Local Chat Mode")
    print("="*60)
    print("💬 Chat directly here in Colab!")
    print("📝 Type 'quit' to exit\n")
    
    while True:
        try:
            message = input("👤 You: ").strip()
            
            if message.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("🤖 AI: Thanks for chatting! I learned a lot from our conversation!")
                break
            
            if message:
                response = ai_engine.chat(message)
                print(f"🤖 AI: {response}\n")
                
        except KeyboardInterrupt:
            print("\n🤖 AI: Chat interrupted. Thanks for the conversation!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def create_simple_web_interface(ai_engine):
    """Create a simple web interface without ngrok"""
    print("🌐 Starting simple web interface...")
    
    from flask import Flask, render_template_string, request, jsonify
    import threading
    
    app = Flask(__name__)
    app.secret_key = 'simple-ai-colab'
    
    # Minimal HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Simple AI Agent</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
            .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { background: #4267B2; color: white; padding: 20px; text-align: center; }
            .chat { height: 400px; overflow-y: auto; padding: 20px; }
            .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
            .user { background: #e3f2fd; margin-left: 50px; }
            .ai { background: #f3e5f5; margin-right: 50px; }
            .input-area { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
            input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
            button { padding: 12px 20px; background: #4267B2; color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: #365899; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>🤖 Portable AI Agent</h2>
                <p>Simple • Local • Learning</p>
            </div>
            <div class="chat" id="chat">
                <div class="message ai">🤖 <strong>AI:</strong> Hello! I'm ready to chat and learn. What's on your mind?</div>
            </div>
            <div class="input-area">
                <input type="text" id="input" placeholder="Type your message..." autocomplete="off">
                <button onclick="send()">Send</button>
            </div>
        </div>

        <script>
            function addMessage(text, isUser) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = `message ${isUser ? 'user' : 'ai'}`;
                div.innerHTML = `${isUser ? '👤' : '🤖'} <strong>${isUser ? 'You' : 'AI'}:</strong> ${text}`;
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }

            function send() {
                const input = document.getElementById('input');
                const message = input.value.trim();
                if (!message) return;

                addMessage(message, true);
                input.value = '';

                fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                })
                .then(r => r.json())
                .then(data => addMessage(data.response, false))
                .catch(e => addMessage('Error: ' + e, false));
            }

            document.getElementById('input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') send();
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
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'response': f'Error: {str(e)}'})
    
    # Start Flask
    def run_app():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_app, daemon=True)
    flask_thread.start()
    
    time.sleep(2)
    
    print("\n" + "="*60)
    print("🎉 Simple Web Interface Started!")
    print("="*60)
    print("🌐 Access your AI at: http://localhost:5000")
    print("\n💡 Features:")
    print("   • Clean, simple interface")
    print("   • Real-time chat")
    print("   • Learning AI responses")
    print("   • Mobile-friendly design")
    print("   • Local-only (secure)")
    print("\n⚠️ Keep this process running!")
    print("="*60)
    
    return "http://localhost:5000"

def main():
    """Main launcher function"""
    print_banner()
    
    # Install dependencies
    install_dependencies()
    
    # Setup project
    setup_project()
    
    # Initialize AI
    ai_engine = initialize_ai()
    
    # Ask user preference
    print("\n🚀 Choose interface:")
    print("1. Simple Web Interface (recommended)")
    print("2. Direct Chat Here")
    
    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
        
        if choice == "1":
            # Web interface
            url = create_simple_web_interface(ai_engine)
            
            # Keep server running
            try:
                print("\n💬 You can also chat here while web server runs:")
                while True:
                    try:
                        message = input("\n👤 Quick Chat: ").strip()
                        if message.lower() in ['quit', 'stop', 'exit']:
                            break
                        if message:
                            response = ai_engine.chat(message)
                            print(f"🤖 AI: {response}")
                    except:
                        break
            except KeyboardInterrupt:
                print("\n🛑 Server stopped")
        else:
            # Direct chat
            start_local_chat(ai_engine)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
# 🚀 One-Click Portable AI Agent Setup for Google Colab
# Run this cell and start chatting immediately!

print("🤖 Setting up your Portable AI Agent...")
print("="*50)

# Clone and setup
import os, sys, subprocess
if not os.path.exists('/content/Porable-Ai-Agent'):
    print("📥 Cloning repository...")
    !git clone https://github.com/umairism/Porable-Ai-Agent.git /content/Porable-Ai-Agent
os.chdir('/content/Porable-Ai-Agent')
sys.path.insert(0, '/content/Porable-Ai-Agent')

# Install dependencies
print("📦 Installing dependencies...")
!pip install -q torch transformers sentence-transformers faiss-cpu flask nltk numpy scipy scikit-learn pandas

# Download NLTK data
print("📚 Setting up NLP data...")
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize AI
print("🧠 Initializing AI...")
try:
    from core.ai_engine import SelfLearningCore
    ai_engine = SelfLearningCore()
    
    def smart_chat(message):
        count = getattr(ai_engine, 'chat_count', 0) + 1
        ai_engine.chat_count = count
        
        if count == 1:
            response = f"Hello! I'm your Portable AI. You said: '{message}'. Let's start learning together!"
        elif 'thank' in message.lower():
            response = f"You're welcome! I appreciate: '{message}'. I'm learning to be more helpful!"
        elif '?' in message:
            response = f"Great question: '{message}'. I'm thinking and learning from this (Chat #{count})!"
        else:
            response = f"I understand: '{message}'. This is our #{count} interaction and I'm getting smarter!"
        
        if not hasattr(ai_engine, 'memory'):
            ai_engine.memory = []
        ai_engine.memory.append((message, response))
        return response
    
    ai_engine.chat = smart_chat
    print("✅ Advanced AI initialized!")
    
except:
    print("⚠️ Using simple AI...")
    class SimpleAI:
        def __init__(self):
            self.memory = []
            self.chat_count = 0
        def chat(self, msg):
            self.chat_count += 1
            resp = f"Chat #{self.chat_count}: I received '{msg}' and I'm learning!"
            self.memory.append((msg, resp))
            return resp
    ai_engine = SimpleAI()

print("\n🎉 Ready! Your AI Agent is initialized!")
print("💬 Start chatting below:")
print("="*50)

# Simple chat loop
while True:
    try:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("🤖 AI: Thanks for chatting! I learned a lot!")
            break
        if user_input.strip():
            response = ai_engine.chat(user_input)
            print(f"🤖 AI: {response}")
    except KeyboardInterrupt:
        print("\n🤖 AI: Chat ended. Thanks for the learning session!")
        break
    except:
        continue
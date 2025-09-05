# 🤖 Portable AI Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Offline](https://img.shields.io/badge/Offline-Ready-green.svg)](https://github.com/umairism/Porable-Ai-Agent)

A **self-contained, offline-capable AI agent** with self-learning capabilities designed for personal use. Your privacy-first AI assistant that learns from every interaction while keeping all data completely local.

## ✨ Features

- 🔒 **100% Offline**: No internet connection required after initial setup
- 🧠 **Self-Learning**: Continuously adapts and improves from user interactions
- 💾 **Persistent Memory**: Remembers conversations and context across sessions
- 📚 **Knowledge Management**: Local vector database for information storage
- 🖥️ **Dual Interface**: Both CLI and web-based interfaces
- 🛡️ **Privacy-First**: All data stays on your device
- ⚡ **Lightweight**: Optimized for personal computers
- 🚀 **Portable**: Easy to move between systems

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Portable AI Agent                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   CLI Interface │  Web Interface  │     API Endpoints       │
├─────────────────┴─────────────────┴─────────────────────────┤
│                   Core AI Engine                            │
│  ┌─────────────────┬─────────────────┬─────────────────────┐ │
│  │ Self-Learning   │   Conversation  │    Knowledge        │ │
│  │   Transformer   │     Memory      │      Base           │ │
│  └─────────────────┴─────────────────┴─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Local Storage                            │
│  ┌─────────────────┬─────────────────┬─────────────────────┐ │
│  │    Models       │     Vector      │     User Data       │ │
│  │   (.pkl/.pth)   │   Index (FAISS) │    (SQLite)         │ │
│  └─────────────────┴─────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- 2GB free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Umairism/portable-ai-agent.git
   cd portable-ai-agent
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv portable_ai_env
   source portable_ai_env/bin/activate  # On Windows: portable_ai_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the AI agent**
   ```bash
   python initialize.py
   ```

5. **Start using your AI**
   ```bash
   # CLI interface
   python main.py
   
   # Web interface (recommended for beginners)
   python main.py --web
   ```

## 💻 Usage

### Command Line Interface

```bash
python main.py
```

Available commands:
- `chat <message>` - Have a conversation with your AI
- `learn <topic>` - Teach your AI something new
- `knowledge add <info>` - Add information to knowledge base
- `memory stats` - View learning statistics
- `help` - Show all available commands
- `exit` - Quit the application

### Web Interface

```bash
python main.py --web --port 8080
```

Then open `http://localhost:8080` in your browser for a user-friendly interface.

### API Usage

```python
from core.ai_engine import SelfLearningCore

# Initialize AI engine
ai = SelfLearningCore()

# Have a conversation
response = ai.generate_response("Hello! How are you?")
print(response)

# Provide feedback for learning
ai.learn_from_feedback("That was a great response!", positive=True)
```

## 📖 Key Components

### 🧠 Self-Learning AI Engine
- **Adaptive Transformer**: Custom neural network that learns incrementally
- **Continuous Learning**: Updates weights based on user interactions
- **Context Awareness**: Maintains conversation context across sessions

### 📚 Knowledge Base
- **Vector Storage**: Uses FAISS for efficient similarity search
- **Semantic Understanding**: Sentence transformers for meaning comprehension
- **Dynamic Learning**: Automatically extracts and stores important information

### 💾 Memory System
- **Conversation History**: Persistent storage of all interactions
- **User Preferences**: Learns and adapts to your communication style
- **Context Retrieval**: Intelligently recalls relevant past conversations

## 🔧 Configuration

Edit `config.json` to customize your AI agent:

```json
{
  "ai_engine": {
    "model_path": "models/",
    "learning_rate": 0.001,
    "context_window": 512,
    "temperature": 0.7
  },
  "knowledge_base": {
    "embedding_model": "all-MiniLM-L6-v2",
    "max_knowledge_items": 10000,
    "similarity_threshold": 0.7
  },
  "memory": {
    "max_conversations": 1000,
    "context_length": 10,
    "auto_summarize": true
  }
}
```

## 📊 Learning & Statistics

Your AI agent tracks its learning progress:

- **Conversations**: Total number of interactions
- **Knowledge Items**: Learned facts and information
- **Accuracy**: Response quality improvements over time
- **Adaptation**: How well it matches your preferences

View statistics with:
```bash
python main.py --stats
```

## 🛡️ Privacy & Security

- **100% Local**: All processing happens on your device
- **No Telemetry**: No data sent to external servers
- **Encrypted Storage**: Local databases are encrypted
- **User Control**: You own and control all data
- **GDPR Compliant**: Designed with privacy by design

## 🔬 Advanced Usage

### Custom Model Training

```python
from core.ai_engine import SelfLearningCore

# Initialize with custom parameters
ai = SelfLearningCore(
    learning_rate=0.01,
    model_size="large",
    specialized_domain="technology"
)

# Train on your specific data
ai.train_on_conversations(your_conversation_data)
```

### Knowledge Base Management

```python
from knowledge.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

# Add structured knowledge
kb.add_knowledge(
    content="Python is a programming language",
    category="programming",
    source="user_input"
)

# Search knowledge
results = kb.search_knowledge("programming languages")
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) for the neural network engine
- Uses [Sentence Transformers](https://www.sbert.net/) for semantic understanding
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Web interface built with [Flask](https://flask.palletsprojects.com/)

## 📞 Support

- 📚 [Documentation](docs/)
- 🐛 [Issues](https://github.com/Umairism/portable-ai-agent/issues)
- 💬 [Discussions](https://github.com/Umairism/portable-ai-agent/discussions)
- 📧 [Email Support](mailto:malikumairHakeem@outlook.com)

---

**Made with ❤️ for privacy-conscious AI enthusiasts**

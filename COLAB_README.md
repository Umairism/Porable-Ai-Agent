# 🤖 Running Portable AI Agent on Google Colab

This guide shows you how to run your Portable AI Agent on Google Colab for free!

## 🚀 Quick Start Methods

### Method 1: Using the Colab Notebook (Recommended)

1. **Open the Notebook**:
   - Click this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/umairism/Porable-Ai-Agent/blob/main/Google_Colab_Portable_AI_Agent.ipynb)

2. **Enable GPU** (Optional but recommended):
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`

3. **Run the Cells**:
   - Run each section in order (Ctrl+Enter)
   - Wait for each section to complete before moving to the next

4. **Access Your AI**:
   - After running Section 3, you'll get a public URL
   - Click the URL to open your AI chat interface
   - Start chatting with your AI!

### Method 2: Using the Python Launcher

1. **Create New Colab Notebook**:
   - Go to [Google Colab](https://colab.research.google.com)
   - Create new notebook

2. **Upload and Run**:
   ```python
   # Upload the colab_launcher.py file to Colab, then run:
   !python colab_launcher.py
   ```

3. **Follow Instructions**:
   - The launcher will automatically setup everything
   - Choose web interface (recommended) or CLI chat
   - Access your AI through the provided URL

### Method 3: Manual Setup

```python
# 1. Clone the repository
!git clone https://github.com/umairism/Porable-Ai-Agent.git
%cd Porable-Ai-Agent

# 2. Install dependencies
!pip install torch transformers sentence-transformers faiss-cpu
!pip install flask nltk numpy scipy scikit-learn pandas
!pip install pyngrok flask-ngrok cryptography

# 3. Setup NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 4. Run the agent
!python main.py --interface web
```

## 🌟 Features Available on Colab

- **🧠 Self-Learning AI**: Learns from every conversation
- **💾 Session Memory**: Remembers context during the session  
- **🚀 GPU Acceleration**: Faster processing with Colab's free GPUs
- **🌐 Web Interface**: Beautiful chat interface accessible via public URL
- **🔒 Secure**: All processing happens in your Colab session
- **📚 Knowledge Base**: Local vector database for information storage
- **💬 Real-time Chat**: Interactive conversation interface

## 📋 System Requirements

- **Free Google Account**: For Google Colab access
- **Internet Connection**: For initial setup and accessing the interface
- **Web Browser**: For the chat interface

### Colab Specifications:
- **RAM**: 12-13 GB available
- **Storage**: ~78 GB temporary storage
- **GPU**: Tesla T4, K80, or P4 (when enabled)
- **Runtime**: Up to 12 hours continuous

## 🔧 Troubleshooting

### Common Issues:

1. **"No module named" errors**:
   ```python
   # Run this to install missing packages:
   !pip install [package-name]
   ```

2. **GPU not available**:
   - Go to Runtime → Change runtime type → GPU
   - Restart runtime and try again

3. **Memory errors**:
   ```python
   # Clear memory:
   import gc
   gc.collect()
   
   # If using GPU:
   import torch
   torch.cuda.empty_cache()
   ```

4. **Web interface not loading**:
   - Try refreshing the ngrok URL
   - Use the CLI chat interface as backup
   - Check if the server is still running

5. **ngrok authentication failed (ERR_NGROK_4018)**:
   - Get free auth token from [ngrok.com](https://ngrok.com)
   - Add `NGROK_AUTH_TOKEN` to Colab secrets (🔑 tab)
   - See [NGROK_SETUP.md](NGROK_SETUP.md) for detailed instructions

5. **Runtime disconnected**:
   - Colab sessions timeout after ~12 hours
   - Simply restart and run the setup again
   - Your conversations are lost unless backed up

### Performance Tips:

- **Enable GPU**: Significantly faster AI processing
- **Close unused tabs**: Saves browser memory
- **Use shorter conversations**: Better for memory management
- **Regular backups**: Save important conversations

## 💾 Data Persistence

⚠️ **Important**: Colab sessions are temporary!

- **Session data** is lost when runtime stops
- **Conversations** are not saved permanently
- **Models and knowledge** reset each session

### Backup Options:

1. **Save to Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Save conversation history
   import pickle
   with open('/content/drive/MyDrive/ai_conversations.pkl', 'wb') as f:
       pickle.dump(ai_engine.memory, f)
   ```

2. **Download Files**:
   ```python
   from google.colab import files
   files.download('conversation_backup.txt')
   ```

## 🔒 Privacy & Security

- **Local Processing**: All AI processing happens in your Colab session
- **No Data Collection**: Your conversations aren't saved by the system
- **Temporary Storage**: Data is automatically deleted when session ends
- **Secure Tunneling**: ngrok provides secure HTTPS access
- **No External APIs**: Everything runs offline within Colab

## 🎯 Use Cases

Perfect for:
- **Testing the AI Agent** before local installation
- **Demonstrations** and showcases
- **Learning AI concepts** with hands-on experience  
- **Prototyping** AI applications
- **Educational purposes** in courses/workshops
- **Quick AI assistance** without setup hassle

## 📚 Additional Resources

- **Main Repository**: [Porable-Ai-Agent](https://github.com/umairism/Porable-Ai-Agent)
- **Documentation**: See main README.md for detailed features
- **Issues**: Report problems on GitHub Issues
- **Discussions**: Join GitHub Discussions for help

## 🤝 Contributing

Found issues or have improvements for the Colab version?
1. Fork the repository
2. Make your changes
3. Submit a pull request
4. Focus on Colab compatibility improvements

## 📄 License

MIT License - See LICENSE file for details

---

## 🎉 Enjoy Your AI Agent on Google Colab!

**Happy chatting and learning! 🤖✨**

Made with ❤️ by [Umair Malik](https://github.com/umairism)
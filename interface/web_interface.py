from flask import Flask, render_template, request, jsonify, session
import os
import sys
import threading
import time
from datetime import datetime
import secrets

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_engine import SelfLearningCore
from knowledge.knowledge_base import KnowledgeBase
from memory.conversation_memory import ConversationMemory


class WebInterface:
    """
    Web-based interface for the Portable AI Agent
    """
    
    def __init__(self, host='127.0.0.1', port=5000, debug=False):
        self.app = Flask(__name__)
        self.app.secret_key = secrets.token_urlsafe(32)
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize AI components
        self.ai_engine = None
        self.knowledge_base = None
        self.memory = None
        self.sessions = {}  # Track user sessions
        
        # Create HTML template
        self.create_html_template()
        
        self.initialize_components()
        self.setup_routes()
        
        # Start auto-save thread
        self.auto_save_active = True
        self.start_auto_save()
    
    def create_html_template(self):
        """Create the HTML template"""
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        # The HTML template is already created as a separate file
        # This method ensures the templates directory exists
        pass

    def initialize_components(self):
        """Initialize AI components"""
        print("Initializing Portable AI Agent for web interface...")
        
        try:
            self.ai_engine = SelfLearningCore()
            self.knowledge_base = KnowledgeBase()
            self.memory = ConversationMemory()
            print("✅ AI Agent web interface initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing AI Agent: {e}")
            raise
    
    def start_auto_save(self):
        """Start background auto-save thread"""
        def auto_save_worker():
            while self.auto_save_active:
                time.sleep(300)  # Save every 5 minutes
                if self.auto_save_active:
                    try:
                        self.ai_engine.save_model()
                        self.knowledge_base.save_knowledge()
                        print("💾 Auto-saved agent state")
                    except Exception as e:
                        print(f"⚠️  Auto-save error: {e}")
        
        auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        auto_save_thread.start()
    
    def get_session_memory(self):
        """Get or create memory instance for current session"""
        session_id = session.get('session_id')
        if not session_id:
            session_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
            session['session_id'] = session_id
        
        if session_id not in self.sessions:
            # Create new memory instance for this session
            memory = ConversationMemory()
            memory.start_session({"interface": "web", "session_id": session_id})
            self.sessions[session_id] = memory
        
        return self.sessions[session_id]
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return render_template('index.html')
        
        @self.app.route('/test')
        def test():
            """Test page"""
            return render_template('test.html')
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Handle chat messages"""
            try:
                data = request.get_json()
                user_message = data.get('message', '').strip()
                
                if not user_message:
                    return jsonify({'error': 'Empty message'}), 400
                
                # Get session memory
                memory = self.get_session_memory()
                
                # Add user message to memory
                memory.add_message("user", user_message, importance=1.0)
                
                # Get relevant context
                context = memory.get_relevant_context(user_message, max_messages=5)
                relevant_knowledge = self.knowledge_base.search_knowledge(user_message, top_k=3)
                
                # Determine AI context
                ai_context = "general"
                if relevant_knowledge:
                    ai_context = relevant_knowledge[0].get('category', 'general')
                
                # Generate response
                response = self.ai_engine.generate_response(user_message, context=ai_context)
                
                # Enhance response with knowledge if relevant
                knowledge_info = None
                if relevant_knowledge and relevant_knowledge[0]['relevance_score'] > 0.7:
                    knowledge_info = {
                        'content': relevant_knowledge[0]['content'][:200],
                        'title': relevant_knowledge[0]['title'],
                        'relevance': relevant_knowledge[0]['relevance_score']
                    }
                
                # Add AI response to memory
                memory.add_message("assistant", response, topic=ai_context, importance=1.0)
                
                # Record interaction
                self.knowledge_base.record_user_interaction(user_message, response, context=ai_context)
                
                # Trigger learning periodically
                if self.ai_engine.performance_metrics['total_interactions'] % 10 == 0:
                    self.ai_engine.continuous_learning()
                
                return jsonify({
                    'response': response,
                    'knowledge_info': knowledge_info,
                    'context': ai_context,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in chat: {e}")
                print(f"Full traceback: {error_details}")
                return jsonify({
                    'error': 'Internal server error', 
                    'details': str(e) if self.debug else 'Check server logs for details'
                }), 500
        
        @self.app.route('/learn', methods=['POST'])
        def learn():
            """Handle learning requests"""
            try:
                data = request.get_json()
                input_text = data.get('input', '').strip()
                expected_output = data.get('output', '').strip()
                score = float(data.get('score', 1.0))
                
                if not input_text or not expected_output:
                    return jsonify({'error': 'Both input and output are required'}), 400
                
                # Train the AI
                self.ai_engine.learn_from_feedback(input_text, expected_output, score)
                
                # Add to knowledge base
                knowledge_id = self.knowledge_base.add_knowledge(
                    content=f"Q: {input_text}\nA: {expected_output}",
                    title=f"User teaching: {input_text[:50]}",
                    category="user_taught",
                    importance=score
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Learning completed successfully',
                    'knowledge_id': knowledge_id
                })
                
            except Exception as e:
                print(f"Error in learn: {e}")
                return jsonify({'error': 'Learning failed'}), 500
        
        @self.app.route('/knowledge', methods=['POST'])
        def add_knowledge():
            """Add knowledge to the system"""
            try:
                data = request.get_json()
                content = data.get('content', '').strip()
                title = data.get('title', '').strip()
                category = data.get('category', 'general').strip()
                
                if not content:
                    return jsonify({'error': 'Content is required'}), 400
                
                if not title:
                    title = f"Knowledge: {content[:50]}"
                
                knowledge_id = self.knowledge_base.add_knowledge(
                    content=content,
                    title=title,
                    category=category,
                    source="web_user"
                )
                
                return jsonify({
                    'success': True,
                    'message': 'Knowledge added successfully',
                    'knowledge_id': knowledge_id
                })
                
            except Exception as e:
                print(f"Error adding knowledge: {e}")
                return jsonify({'error': 'Failed to add knowledge'}), 500
        
        @self.app.route('/search', methods=['POST'])
        def search():
            """Search knowledge base"""
            try:
                data = request.get_json()
                query = data.get('query', '').strip()
                top_k = int(data.get('top_k', 5))
                
                if not query:
                    return jsonify({'error': 'Query is required'}), 400
                
                results = self.knowledge_base.search_knowledge(query, top_k=top_k)
                
                return jsonify({
                    'results': results,
                    'total': len(results)
                })
                
            except Exception as e:
                print(f"Error in search: {e}")
                return jsonify({'error': 'Search failed'}), 500
        
        @self.app.route('/stats')
        def stats():
            """Get system statistics"""
            try:
                ai_stats = self.ai_engine.get_performance_stats()
                kb_stats = self.knowledge_base.get_statistics()
                
                # Get memory stats for current session
                memory = self.get_session_memory()
                memory_stats = memory.get_conversation_statistics()
                
                return jsonify({
                    'ai_engine': ai_stats,
                    'knowledge_base': kb_stats,
                    'memory': memory_stats,
                    'active_sessions': len(self.sessions)
                })
                
            except Exception as e:
                print(f"Error getting stats: {e}")
                return jsonify({'error': 'Failed to get statistics'}), 500
        
        @self.app.route('/insights')
        def insights():
            """Get learning insights"""
            try:
                insights = self.knowledge_base.learn_from_interactions()
                return jsonify(insights)
                
            except Exception as e:
                print(f"Error getting insights: {e}")
                return jsonify({'error': 'Failed to get insights'}), 500
        
        @self.app.route('/save', methods=['POST'])
        def save():
            """Save all system data"""
            try:
                self.ai_engine.save_model()
                self.knowledge_base.save_knowledge()
                
                return jsonify({
                    'success': True,
                    'message': 'All data saved successfully',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Error saving: {e}")
                return jsonify({'error': 'Save failed'}), 500
    
    def run(self):
        """Run the web interface"""
        print(f"🌐 Starting Portable AI Agent Web Interface...")
        print(f"🔗 Access at: http://{self.host}:{self.port}")
        print(f"🔒 All processing happens locally - your data stays private")
        
        try:
            self.app.run(host=self.host, port=self.port, debug=self.debug, threaded=True)
        finally:
            self.auto_save_active = False
            
            # Final save
            try:
                self.ai_engine.save_model()
                self.knowledge_base.save_knowledge()
                print("💾 Final save completed")
            except Exception as e:
                print(f"⚠️  Error during final save: {e}")


def create_html_template():
    """Create the HTML template"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portable AI Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            width: 90%;
            max-width: 800px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
            max-height: 400px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .ai-message {
            background: #e9ecef;
            color: #333;
            margin-right: auto;
        }
        
        .knowledge-info {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 8px;
            margin-top: 8px;
            font-size: 12px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .input-container input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-container input:focus {
            border-color: #667eea;
        }
        
        .input-container button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: transform 0.2s;
        }
        
        .input-container button:hover {
            transform: translateY(-2px);
        }
        
        .input-container button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-radius: 10px 10px 0 0;
            margin-top: 20px;
        }
        
        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab.active {
            background: white;
            border-bottom-color: #667eea;
            color: #667eea;
        }
        
        .tab-content {
            display: none;
            padding: 20px;
            background: white;
            border-radius: 0 0 10px 10px;
            border: 1px solid #e9ecef;
            border-top: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .form-group textarea {
            height: 100px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-1px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px;
            font-style: italic;
            color: #666;
        }
        
        .typing-indicator.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Portable AI Agent</h1>
            <p>Offline • Self-Learning • Private</p>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message ai-message">
                    <div>👋 Hello! I'm your personal AI assistant. I learn from our conversations and work completely offline. How can I help you today?</div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                🤔 AI is thinking...
            </div>
            
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type your message here..." 
                       onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 Chat</button>
            <button class="tab" onclick="showTab('learn')">📚 Learn</button>
            <button class="tab" onclick="showTab('knowledge')">🧠 Knowledge</button>
            <button class="tab" onclick="showTab('stats')">📊 Stats</button>
        </div>
        
        <div id="chat-content" class="tab-content active">
            <!-- Chat is above in the main container -->
        </div>
        
        <div id="learn-content" class="tab-content">
            <h3>Teach the AI</h3>
            <div class="form-group">
                <label>Question/Input:</label>
                <input type="text" id="learnInput" placeholder="What should the AI know?">
            </div>
            <div class="form-group">
                <label>Expected Response:</label>
                <textarea id="learnOutput" placeholder="How should the AI respond?"></textarea>
            </div>
            <div class="form-group">
                <label>Quality Score (0-1):</label>
                <input type="number" id="learnScore" min="0" max="1" step="0.1" value="1.0">
            </div>
            <button class="btn" onclick="teachAI()">Teach AI</button>
        </div>
        
        <div id="knowledge-content" class="tab-content">
            <h3>Add Knowledge</h3>
            <div class="form-group">
                <label>Content:</label>
                <textarea id="knowledgeContent" placeholder="Enter knowledge content..."></textarea>
            </div>
            <div class="form-group">
                <label>Title:</label>
                <input type="text" id="knowledgeTitle" placeholder="Knowledge title (optional)">
            </div>
            <div class="form-group">
                <label>Category:</label>
                <input type="text" id="knowledgeCategory" placeholder="Category (e.g., general, technical)">
            </div>
            <button class="btn" onclick="addKnowledge()">Add Knowledge</button>
            
            <hr style="margin: 20px 0;">
            
            <h3>Search Knowledge</h3>
            <div class="form-group">
                <input type="text" id="searchQuery" placeholder="Search knowledge base...">
                <button class="btn" onclick="searchKnowledge()" style="margin-top: 10px;">Search</button>
            </div>
            <div id="searchResults"></div>
        </div>
        
        <div id="stats-content" class="tab-content">
            <h3>AI Statistics</h3>
            <div id="statsContainer">
                <button class="btn" onclick="loadStats()">Refresh Stats</button>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            if (tabName === 'chat') {
                document.getElementById('chat-content').classList.add('active');
                document.querySelector('[onclick="showTab(\'chat\')"]').classList.add('active');
            } else {
                document.getElementById(tabName + '-content').classList.add('active');
                document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            input.value = '';
            
            // Show typing indicator
            document.getElementById('typingIndicator').classList.add('show');
            document.getElementById('sendButton').disabled = true;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('ai', 'Sorry, I encountered an error: ' + data.error);
                } else {
                    addMessage('ai', data.response, data.knowledge_info);
                }
                
            } catch (error) {
                addMessage('ai', 'Sorry, I encountered a connection error. Please try again.');
            } finally {
                document.getElementById('typingIndicator').classList.remove('show');
                document.getElementById('sendButton').disabled = false;
                input.focus();
            }
        }
        
        function addMessage(role, content, knowledgeInfo = null) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}-message`;
            
            let messageHTML = `<div>${content}</div>`;
            
            if (knowledgeInfo) {
                messageHTML += `<div class="knowledge-info">
                    💡 <strong>${knowledgeInfo.title}</strong><br>
                    ${knowledgeInfo.content}... (Relevance: ${(knowledgeInfo.relevance * 100).toFixed(1)}%)
                </div>`;
            }
            
            messageDiv.innerHTML = messageHTML;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        async function teachAI() {
            const input = document.getElementById('learnInput').value.trim();
            const output = document.getElementById('learnOutput').value.trim();
            const score = parseFloat(document.getElementById('learnScore').value);
            
            if (!input || !output) {
                alert('Please provide both input and expected output');
                return;
            }
            
            try {
                const response = await fetch('/learn', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input, output, score })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ AI learning completed!');
                    document.getElementById('learnInput').value = '';
                    document.getElementById('learnOutput').value = '';
                } else {
                    alert('❌ Learning failed: ' + data.error);
                }
                
            } catch (error) {
                alert('❌ Connection error during learning');
            }
        }
        
        async function addKnowledge() {
            const content = document.getElementById('knowledgeContent').value.trim();
            const title = document.getElementById('knowledgeTitle').value.trim();
            const category = document.getElementById('knowledgeCategory').value.trim() || 'general';
            
            if (!content) {
                alert('Please provide knowledge content');
                return;
            }
            
            try {
                const response = await fetch('/knowledge', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content, title, category })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ Knowledge added successfully!');
                    document.getElementById('knowledgeContent').value = '';
                    document.getElementById('knowledgeTitle').value = '';
                    document.getElementById('knowledgeCategory').value = '';
                } else {
                    alert('❌ Failed to add knowledge: ' + data.error);
                }
                
            } catch (error) {
                alert('❌ Connection error while adding knowledge');
            }
        }
        
        async function searchKnowledge() {
            const query = document.getElementById('searchQuery').value.trim();
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                const data = await response.json();
                const resultsDiv = document.getElementById('searchResults');
                
                if (data.results && data.results.length > 0) {
                    let html = '<h4>Search Results:</h4>';
                    data.results.forEach((result, index) => {
                        html += `
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 6px;">
                                <strong>${result.title}</strong> (${result.category})<br>
                                <small>Relevance: ${(result.relevance_score * 100).toFixed(1)}%</small><br>
                                ${result.content.substring(0, 200)}...
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<p>No results found.</p>';
                }
                
            } catch (error) {
                document.getElementById('searchResults').innerHTML = '<p>Search error occurred.</p>';
            }
        }
        
        async function loadStats() {
            const container = document.getElementById('statsContainer');
            container.innerHTML = '<p>Loading statistics...</p>';
            
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                let html = '<div class="stats-grid">';
                
                // AI Engine stats
                html += `
                    <div class="stat-card">
                        <div class="stat-value">${data.ai_engine.total_interactions}</div>
                        <div class="stat-label">Total Interactions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.ai_engine.learning_events}</div>
                        <div class="stat-label">Learning Events</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${(data.ai_engine.success_rate * 100).toFixed(1)}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.knowledge_base.total_knowledge_items}</div>
                        <div class="stat-label">Knowledge Items</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.memory.total_messages}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.active_sessions}</div>
                        <div class="stat-label">Active Sessions</div>
                    </div>
                `;
                
                html += '</div>';
                html += '<button class="btn" onclick="loadStats()" style="margin-top: 20px;">Refresh</button>';
                
                container.innerHTML = html;
                
            } catch (error) {
                container.innerHTML = '<p>Error loading statistics.</p>';
            }
        }
        
        // Load initial stats when stats tab is first shown
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('messageInput').focus();
        });
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'index.html'), 'w') as f:
        f.write(html_content)


def main():
    """Main entry point for web interface"""
    # Create HTML template
    create_html_template()
    
    # Start web interface
    web_interface = WebInterface(host='127.0.0.1', port=5000, debug=False)
    web_interface.run()


if __name__ == "__main__":
    main()

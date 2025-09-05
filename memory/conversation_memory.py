import sqlite3
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import numpy as np
from collections import defaultdict, deque
import threading
import time


class ConversationMemory:
    """
    Manages conversation history and contextual memory for the AI agent
    """
    
    def __init__(self, db_path: str = "memory/", max_context_length: int = 10000):
        self.db_path = db_path
        self.max_context_length = max_context_length
        
        os.makedirs(db_path, exist_ok=True)
        
        # Database for persistent storage
        self.db_file = os.path.join(db_path, "memory.db")
        self.init_database()
        
        # In-memory conversation context
        self.current_context = deque(maxlen=50)  # Keep last 50 exchanges
        self.session_id = self.generate_session_id()
        
        # User profile and preferences
        self.user_profile = self.load_user_profile()
        
        # Conversation patterns and learning
        self.conversation_patterns = defaultdict(list)
        self.topic_transitions = defaultdict(int)
        
        # Load recent context
        self.load_recent_context()
    
    def init_database(self):
        """Initialize memory database schema"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Conversation sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                topics TEXT,
                user_satisfaction REAL,
                metadata TEXT
            )
        ''')
        
        # Individual messages
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                topic TEXT,
                sentiment REAL,
                importance REAL DEFAULT 1.0,
                context_embedding BLOB,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions (id)
            )
        ''')
        
        # User preferences and profile
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT,
                category TEXT,
                confidence REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Learning patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER DEFAULT 1,
                effectiveness REAL DEFAULT 0.5,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Long-term memory summaries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_period TEXT,
                summary TEXT,
                key_topics TEXT,
                important_facts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    def start_session(self, metadata: Dict = None):
        """Start a new conversation session"""
        self.session_id = self.generate_session_id()
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_sessions (id, metadata)
            VALUES (?, ?)
        ''', (self.session_id, json.dumps(metadata) if metadata else None))
        
        conn.commit()
        conn.close()
        
        # Clear current context for new session
        self.current_context.clear()
        
        print(f"Started new conversation session: {self.session_id}")
    
    def add_message(self, role: str, content: str, topic: str = None, 
                   sentiment: float = None, importance: float = 1.0, 
                   metadata: Dict = None):
        """Add a message to conversation memory"""
        
        # Add to current context
        message_data = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'sentiment': sentiment,
            'importance': importance,
            'metadata': metadata
        }
        self.current_context.append(message_data)
        
        # Store in database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Simple context embedding (in practice, use proper embeddings)
        context_embedding = self.generate_context_embedding(content)
        
        cursor.execute('''
            INSERT INTO messages 
            (session_id, role, content, topic, sentiment, importance, context_embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.session_id, role, content, topic, sentiment, importance,
              pickle.dumps(context_embedding), json.dumps(metadata) if metadata else None))
        
        # Update session message count
        cursor.execute('''
            UPDATE conversation_sessions 
            SET message_count = message_count + 1
            WHERE id = ?
        ''', (self.session_id,))
        
        conn.commit()
        conn.close()
        
        # Learn from conversation patterns
        if len(self.current_context) >= 2:
            self.learn_conversation_pattern()
    
    def generate_context_embedding(self, text: str) -> np.ndarray:
        """Generate simple context embedding"""
        # Simplified embedding - in practice, use proper sentence embeddings
        words = text.lower().split()
        # Create a simple bag-of-words style embedding
        embedding = np.zeros(100)
        for i, word in enumerate(words[:10]):  # Take first 10 words
            word_hash = hash(word) % 100
            embedding[word_hash] += 1
        
        # Normalize
        if np.sum(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
    
    def get_relevant_context(self, query: str, max_messages: int = 10) -> List[Dict]:
        """Get relevant conversation context for a query"""
        
        # Search in current context first
        relevant_context = []
        query_embedding = self.generate_context_embedding(query)
        
        # Score current context messages
        context_scores = []
        for msg in self.current_context:
            msg_embedding = self.generate_context_embedding(msg['content'])
            similarity = np.dot(query_embedding, msg_embedding)
            context_scores.append((similarity, msg))
        
        # Sort by relevance and importance
        context_scores.sort(key=lambda x: x[0] * x[1].get('importance', 1.0), reverse=True)
        
        # Add top relevant messages from current context
        for score, msg in context_scores[:max_messages//2]:
            if score > 0.1:  # Minimum relevance threshold
                relevant_context.append(msg)
        
        # Search historical messages if needed
        if len(relevant_context) < max_messages:
            historical_context = self.search_historical_context(query, max_messages - len(relevant_context))
            relevant_context.extend(historical_context)
        
        return relevant_context
    
    def search_historical_context(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search historical conversation context"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Simple text search (in practice, use vector similarity)
        query_words = query.lower().split()
        search_conditions = []
        params = []
        
        for word in query_words[:3]:  # Use first 3 words
            search_conditions.append("LOWER(content) LIKE ?")
            params.append(f"%{word}%")
        
        where_clause = " OR ".join(search_conditions) if search_conditions else "1=1"
        
        cursor.execute(f'''
            SELECT role, content, topic, sentiment, importance, timestamp, metadata
            FROM messages
            WHERE ({where_clause}) AND session_id != ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        ''', params + [self.session_id, max_results])
        
        results = []
        for row in cursor.fetchall():
            role, content, topic, sentiment, importance, timestamp, metadata = row
            results.append({
                'role': role,
                'content': content,
                'topic': topic,
                'sentiment': sentiment,
                'importance': importance,
                'timestamp': timestamp,
                'metadata': json.loads(metadata) if metadata else None
            })
        
        conn.close()
        return results
    
    def learn_conversation_pattern(self):
        """Learn from current conversation patterns"""
        if len(self.current_context) < 2:
            return
        
        # Analyze recent message pair
        current_msg = self.current_context[-1]
        previous_msg = self.current_context[-2]
        
        # Learn topic transitions
        if current_msg.get('topic') and previous_msg.get('topic'):
            transition = f"{previous_msg['topic']} -> {current_msg['topic']}"
            self.topic_transitions[transition] += 1
        
        # Learn response patterns
        if previous_msg['role'] == 'user' and current_msg['role'] == 'assistant':
            pattern_key = f"response_to_{previous_msg.get('topic', 'general')}"
            self.conversation_patterns[pattern_key].append({
                'input': previous_msg['content'][:100],  # First 100 chars
                'output': current_msg['content'][:100],
                'effectiveness': current_msg.get('importance', 1.0)
            })
        
        # Store learned patterns periodically
        if len(self.conversation_patterns) % 10 == 0:
            self.store_learned_patterns()
    
    def store_learned_patterns(self):
        """Store learned conversation patterns"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        for pattern_type, patterns in self.conversation_patterns.items():
            pattern_data = json.dumps(patterns[-5:])  # Keep last 5 patterns
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns 
                (pattern_type, pattern_data, frequency, effectiveness)
                VALUES (?, ?, ?, ?)
            ''', (pattern_type, pattern_data, len(patterns), 
                  np.mean([p.get('effectiveness', 0.5) for p in patterns[-5:]])))
        
        # Store topic transitions
        if self.topic_transitions:
            transition_data = json.dumps(dict(self.topic_transitions))
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns 
                (pattern_type, pattern_data, frequency)
                VALUES (?, ?, ?)
            ''', ('topic_transitions', transition_data, sum(self.topic_transitions.values())))
        
        conn.commit()
        conn.close()
        
        # Clear old patterns to prevent memory bloat
        self.conversation_patterns.clear()
        self.topic_transitions.clear()
    
    def update_user_profile(self, key: str, value: str, category: str = "general", 
                           confidence: float = 1.0):
        """Update user profile information"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_profile 
            (key, value, category, confidence, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (key, value, category, confidence))
        
        conn.commit()
        conn.close()
        
        # Update in-memory profile
        self.user_profile[key] = {
            'value': value,
            'category': category,
            'confidence': confidence
        }
    
    def get_user_preference(self, key: str, default=None):
        """Get user preference from profile"""
        if key in self.user_profile:
            return self.user_profile[key]['value']
        return default
    
    def load_user_profile(self) -> Dict:
        """Load user profile from database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT key, value, category, confidence FROM user_profile')
        
        profile = {}
        for key, value, category, confidence in cursor.fetchall():
            profile[key] = {
                'value': value,
                'category': category,
                'confidence': confidence
            }
        
        conn.close()
        return profile
    
    def load_recent_context(self, days: int = 7):
        """Load recent conversation context"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT role, content, topic, sentiment, importance, timestamp, metadata
            FROM messages
            WHERE timestamp > ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        ''', (cutoff_date.isoformat(), self.session_id))
        
        for row in cursor.fetchall():
            role, content, topic, sentiment, importance, timestamp, metadata = row
            message_data = {
                'role': role,
                'content': content,
                'topic': topic,
                'sentiment': sentiment,
                'importance': importance,
                'timestamp': timestamp,
                'metadata': json.loads(metadata) if metadata else None
            }
            self.current_context.appendleft(message_data)  # Add to beginning
        
        conn.close()
    
    def create_memory_summary(self, time_period: str = "daily"):
        """Create a summary of conversations for long-term memory"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Define time range based on period
        if time_period == "daily":
            start_time = datetime.now() - timedelta(days=1)
        elif time_period == "weekly":
            start_time = datetime.now() - timedelta(weeks=1)
        elif time_period == "monthly":
            start_time = datetime.now() - timedelta(days=30)
        else:
            start_time = datetime.now() - timedelta(days=1)
        
        # Get messages from time period
        cursor.execute('''
            SELECT content, topic, importance FROM messages
            WHERE timestamp > ? AND role = 'user'
            ORDER BY importance DESC
        ''', (start_time.isoformat(),))
        
        user_messages = cursor.fetchall()
        
        if not user_messages:
            conn.close()
            return
        
        # Extract key information
        topics = defaultdict(int)
        important_facts = []
        
        for content, topic, importance in user_messages:
            if topic:
                topics[topic] += 1
            if importance > 1.5:  # High importance threshold
                important_facts.append(content[:200])  # First 200 chars
        
        # Create summary
        summary_parts = []
        if topics:
            main_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts.append(f"Main topics: {', '.join([t[0] for t in main_topics])}")
        
        if important_facts:
            summary_parts.append(f"Important discussions: {len(important_facts)} key points")
        
        summary = "; ".join(summary_parts) if summary_parts else "No significant activity"
        
        # Store summary
        cursor.execute('''
            INSERT INTO memory_summaries 
            (time_period, summary, key_topics, important_facts)
            VALUES (?, ?, ?, ?)
        ''', (time_period, summary, json.dumps(dict(topics)), json.dumps(important_facts)))
        
        conn.commit()
        conn.close()
        
        print(f"Created {time_period} memory summary: {summary}")
    
    def get_conversation_statistics(self) -> Dict:
        """Get conversation statistics"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM messages')
        total_messages = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversation_sessions')
        total_sessions = cursor.fetchone()[0]
        
        # Topic distribution
        cursor.execute('SELECT topic, COUNT(*) FROM messages WHERE topic IS NOT NULL GROUP BY topic')
        topics = dict(cursor.fetchall())
        
        # Recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM messages 
            WHERE timestamp > datetime('now', '-7 days')
        ''')
        recent_messages = cursor.fetchone()[0]
        
        # User profile size
        cursor.execute('SELECT COUNT(*) FROM user_profile')
        profile_entries = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_messages': total_messages,
            'total_sessions': total_sessions,
            'topics': topics,
            'recent_messages_7d': recent_messages,
            'user_profile_entries': profile_entries,
            'current_context_size': len(self.current_context),
            'current_session_id': self.session_id
        }
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old conversation data to manage storage"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Delete old messages (but keep summaries)
        cursor.execute('''
            DELETE FROM messages 
            WHERE timestamp < ? AND importance < 1.5
        ''', (cutoff_date.isoformat(),))
        
        deleted_messages = cursor.rowcount
        
        # Delete old sessions with no remaining messages
        cursor.execute('''
            DELETE FROM conversation_sessions 
            WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)
        ''')
        
        deleted_sessions = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"Cleaned up {deleted_messages} old messages and {deleted_sessions} empty sessions")
        return deleted_messages, deleted_sessions

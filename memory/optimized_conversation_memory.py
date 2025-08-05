"""
Optimized Conversation Memory with Performance Improvements
Addresses database I/O bottlenecks and memory inefficiencies.
"""

import sqlite3
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import threading
import asyncio
import time
import hashlib
from collections import defaultdict, deque
from functools import lru_cache
import logging
import os


class OptimizedConversationMemory:
    """
    High-performance conversation memory with batching, caching, and async operations
    """
    
    def __init__(self, db_path: str = "data/conversations.db", 
                 cache_size: int = 5000, batch_size: int = 25):
        
        self.db_path = db_path
        self.cache_size = cache_size
        self.batch_size = batch_size
        
        # Initialize database with optimizations
        self.init_database()
        
        # In-memory caches
        self.conversation_cache = {}  # Recent conversations
        self.pattern_cache = {}       # Learned patterns
        self.user_profile_cache = {}  # User profiles
        self.session_cache = {}       # Current session data
        
        # Batch processing
        self.pending_messages = deque()
        self.pending_patterns = deque()
        self.auto_batch_enabled = True
        
        # Threading and async
        self.lock = threading.RLock()
        self.background_tasks = []
        
        # Current session tracking
        self.current_session = {
            'session_id': self.generate_session_id(),
            'user_id': 'default_user',
            'start_time': datetime.now(),
            'message_count': 0,
            'context_summary': "",
            'conversation_buffer': deque(maxlen=50)  # Keep last 50 messages in memory
        }
        
        # Performance metrics
        self.metrics = {
            'total_messages': 0,
            'cache_hits': 0,
            'batch_writes': 0,
            'pattern_discoveries': 0,
            'average_write_time': 0.0,
            'session_count': 0
        }
        
        # Pattern learning
        self.pattern_learner = ConversationPatternLearner()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self.load_user_profiles()
    
    def init_database(self):
        """Initialize optimized SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=20000")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
            
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,  -- 'user', 'assistant'
                    content TEXT NOT NULL,
                    content_hash TEXT,
                    context_tags TEXT DEFAULT '[]',
                    sentiment_score REAL DEFAULT 0.0,
                    importance_score REAL DEFAULT 1.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_time_ms INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Optimized indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_time ON conversations(session_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_time ON conversations(user_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON conversations(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON conversations(importance_score DESC)")
            
            # User profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    preferences TEXT DEFAULT '{}',
                    conversation_style TEXT DEFAULT 'adaptive',
                    learning_history TEXT DEFAULT '[]',
                    interaction_count INTEGER DEFAULT 0,
                    first_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    profile_data TEXT DEFAULT '{}'
                )
            """)
            
            # Conversation patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_signature TEXT NOT NULL UNIQUE,
                    pattern_data TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    effectiveness_score REAL DEFAULT 1.0,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON conversation_patterns(pattern_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_user ON conversation_patterns(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_frequency ON conversation_patterns(frequency DESC)")
            
            # Session summaries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    summary TEXT,
                    key_topics TEXT DEFAULT '[]',
                    sentiment_trend TEXT DEFAULT 'neutral',
                    session_quality REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Context embeddings table (for similarity search)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    context_window TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversations (session_id)
                )
            """)
            
            conn.commit()
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = int(time.time() * 1000)
        return f"session_{timestamp}_{hash(str(timestamp)) % 10000:04d}"
    
    def get_content_hash(self, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def add_message(self, content: str, message_type: str = "user", 
                   context_tags: List[str] = None, importance_score: float = 1.0,
                   response_time_ms: int = 0, metadata: Dict = None, 
                   batch_mode: bool = None) -> Optional[int]:
        """Add message with optional batching"""
        if batch_mode is None:
            batch_mode = self.auto_batch_enabled
        
        content_hash = self.get_content_hash(content)
        
        message_data = {
            'session_id': self.current_session['session_id'],
            'user_id': self.current_session['user_id'],
            'message_type': message_type,
            'content': content,
            'content_hash': content_hash,
            'context_tags': json.dumps(context_tags or []),
            'importance_score': importance_score,
            'response_time_ms': response_time_ms,
            'metadata': json.dumps(metadata or {}),
            'timestamp': datetime.now()
        }
        
        # Add to current session buffer
        self.current_session['conversation_buffer'].append(message_data)
        self.current_session['message_count'] += 1
        
        # Update session cache
        session_key = f"{self.current_session['user_id']}_{self.current_session['session_id']}"
        if session_key not in self.session_cache:
            self.session_cache[session_key] = []
        self.session_cache[session_key].append(message_data)
        
        if batch_mode:
            # Add to pending batch
            self.pending_messages.append(message_data)
            
            # Process batch if full
            if len(self.pending_messages) >= self.batch_size:
                self.process_message_batch()
            
            return None  # Will get ID after batch processing
        else:
            # Process immediately
            return self._add_message_immediate(message_data)
    
    def _add_message_immediate(self, message_data: Dict) -> Optional[int]:
        """Add single message immediately"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO conversations 
                    (session_id, user_id, message_type, content, content_hash, 
                     context_tags, importance_score, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message_data['session_id'],
                    message_data['user_id'],
                    message_data['message_type'],
                    message_data['content'],
                    message_data['content_hash'],
                    message_data['context_tags'],
                    message_data['importance_score'],
                    message_data['response_time_ms'],
                    message_data['metadata']
                ))
                
                message_id = cursor.lastrowid
                self.metrics['total_messages'] += 1
                
                # Update user profile
                self.update_user_profile(message_data['user_id'])
                
                return message_id
                
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            return None
    
    def process_message_batch(self) -> List[int]:
        """Process pending messages in batch"""
        if not self.pending_messages:
            return []
        
        start_time = time.time()
        batch = list(self.pending_messages)
        self.pending_messages.clear()
        
        added_ids = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare batch data
                batch_data = [
                    (
                        msg['session_id'], msg['user_id'], msg['message_type'],
                        msg['content'], msg['content_hash'], msg['context_tags'],
                        msg['importance_score'], msg['response_time_ms'], msg['metadata']
                    )
                    for msg in batch
                ]
                
                # Batch insert
                cursor = conn.executemany("""
                    INSERT INTO conversations 
                    (session_id, user_id, message_type, content, content_hash, 
                     context_tags, importance_score, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                
                # Get inserted IDs
                first_id = cursor.lastrowid
                added_ids = list(range(first_id, first_id + len(batch_data)))
                
                self.metrics['total_messages'] += len(batch_data)
                self.metrics['batch_writes'] += 1
                
                # Update user profiles for unique users in batch
                unique_users = set(msg['user_id'] for msg in batch)
                for user_id in unique_users:
                    self.update_user_profile(user_id)
                
                write_time = time.time() - start_time
                self.metrics['average_write_time'] = (
                    (self.metrics['average_write_time'] * (self.metrics['batch_writes'] - 1) + write_time) /
                    self.metrics['batch_writes']
                )
                
                self.logger.info(f"Batch processed: {len(batch_data)} messages in {write_time:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            # Re-add failed messages to pending queue
            self.pending_messages.extend(batch)
        
        return added_ids
    
    @lru_cache(maxsize=1000)
    def get_conversation_history_cached(self, cache_key: str, user_id: str, 
                                       limit: int = 50, session_id: str = None) -> List[Dict]:
        """Get conversation history with caching"""
        return self._get_conversation_history_internal(user_id, limit, session_id)
    
    def _get_conversation_history_internal(self, user_id: str, limit: int = 50, 
                                          session_id: str = None) -> List[Dict]:
        """Internal conversation history retrieval"""
        try:
            # Check session cache first
            if session_id:
                session_key = f"{user_id}_{session_id}"
                if session_key in self.session_cache:
                    cached_messages = self.session_cache[session_key][-limit:]
                    return [self._format_message_dict(msg) for msg in cached_messages]
            
            # Query database
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    cursor = conn.execute("""
                        SELECT * FROM conversations 
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (user_id, session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM conversations 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (user_id, limit))
                
                messages = []
                for row in cursor.fetchall():
                    message = {
                        'id': row[0],
                        'session_id': row[1],
                        'user_id': row[2],
                        'message_type': row[3],
                        'content': row[4],
                        'context_tags': json.loads(row[6] or '[]'),
                        'sentiment_score': row[7],
                        'importance_score': row[8],
                        'timestamp': row[9],
                        'response_time_ms': row[10],
                        'metadata': json.loads(row[11] or '{}')
                    }
                    messages.append(message)
                
                # Reverse to get chronological order
                return list(reversed(messages))
                
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []
    
    def _format_message_dict(self, msg_data: Dict) -> Dict:
        """Format message data for consistent output"""
        return {
            'session_id': msg_data['session_id'],
            'user_id': msg_data['user_id'],
            'message_type': msg_data['message_type'],
            'content': msg_data['content'],
            'context_tags': json.loads(msg_data['context_tags']) if isinstance(msg_data['context_tags'], str) else msg_data['context_tags'],
            'importance_score': msg_data['importance_score'],
            'timestamp': msg_data['timestamp'],
            'metadata': json.loads(msg_data['metadata']) if isinstance(msg_data['metadata'], str) else msg_data['metadata']
        }
    
    def get_conversation_history(self, user_id: str = None, limit: int = 50, 
                               session_id: str = None) -> List[Dict]:
        """Get conversation history with caching"""
        if user_id is None:
            user_id = self.current_session['user_id']
        
        # Generate cache key
        cache_key = f"{user_id}_{limit}_{session_id or 'all'}"
        
        return self.get_conversation_history_cached(cache_key, user_id, limit, session_id)
    
    async def get_conversation_history_async(self, user_id: str = None, limit: int = 50, 
                                           session_id: str = None) -> List[Dict]:
        """Async conversation history retrieval"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get_conversation_history, user_id, limit, session_id
        )
    
    def learn_conversation_pattern(self, pattern_type: str, pattern_data: Dict, 
                                 effectiveness_score: float = 1.0):
        """Learn conversation pattern with batching"""
        pattern_signature = self.generate_pattern_signature(pattern_type, pattern_data)
        
        pattern_entry = {
            'pattern_type': pattern_type,
            'pattern_signature': pattern_signature,
            'pattern_data': json.dumps(pattern_data),
            'effectiveness_score': effectiveness_score,
            'user_id': self.current_session['user_id'],
            'timestamp': datetime.now()
        }
        
        if self.auto_batch_enabled:
            self.pending_patterns.append(pattern_entry)
            
            if len(self.pending_patterns) >= self.batch_size // 2:  # Smaller batch for patterns
                self.process_pattern_batch()
        else:
            self._learn_pattern_immediate(pattern_entry)
    
    def _learn_pattern_immediate(self, pattern_entry: Dict):
        """Learn single pattern immediately"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if pattern exists
                cursor = conn.execute("""
                    SELECT id, frequency FROM conversation_patterns 
                    WHERE pattern_signature = ?
                """, (pattern_entry['pattern_signature'],))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update frequency and effectiveness
                    new_frequency = existing[1] + 1
                    conn.execute("""
                        UPDATE conversation_patterns 
                        SET frequency = ?, 
                            effectiveness_score = (effectiveness_score + ?) / 2,
                            last_seen = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_frequency, pattern_entry['effectiveness_score'], existing[0]))
                else:
                    # Insert new pattern
                    conn.execute("""
                        INSERT INTO conversation_patterns 
                        (pattern_type, pattern_signature, pattern_data, 
                         effectiveness_score, user_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        pattern_entry['pattern_type'],
                        pattern_entry['pattern_signature'],
                        pattern_entry['pattern_data'],
                        pattern_entry['effectiveness_score'],
                        pattern_entry['user_id']
                    ))
                    
                    self.metrics['pattern_discoveries'] += 1
                
        except Exception as e:
            self.logger.error(f"Error learning pattern: {e}")
    
    def process_pattern_batch(self):
        """Process pending patterns in batch"""
        if not self.pending_patterns:
            return
        
        batch = list(self.pending_patterns)
        self.pending_patterns.clear()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for pattern_entry in batch:
                    # Check if pattern exists
                    cursor = conn.execute("""
                        SELECT id, frequency FROM conversation_patterns 
                        WHERE pattern_signature = ?
                    """, (pattern_entry['pattern_signature'],))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing
                        new_frequency = existing[1] + 1
                        conn.execute("""
                            UPDATE conversation_patterns 
                            SET frequency = ?, 
                                effectiveness_score = (effectiveness_score + ?) / 2,
                                last_seen = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (new_frequency, pattern_entry['effectiveness_score'], existing[0]))
                    else:
                        # Insert new
                        conn.execute("""
                            INSERT INTO conversation_patterns 
                            (pattern_type, pattern_signature, pattern_data, 
                             effectiveness_score, user_id)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            pattern_entry['pattern_type'],
                            pattern_entry['pattern_signature'],
                            pattern_entry['pattern_data'],
                            pattern_entry['effectiveness_score'],
                            pattern_entry['user_id']
                        ))
                        
                        self.metrics['pattern_discoveries'] += 1
                
                self.logger.info(f"Pattern batch processed: {len(batch)} patterns")
                
        except Exception as e:
            self.logger.error(f"Error in pattern batch processing: {e}")
            self.pending_patterns.extend(batch)
    
    def generate_pattern_signature(self, pattern_type: str, pattern_data: Dict) -> str:
        """Generate unique signature for pattern"""
        data_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(f"{pattern_type}_{data_str}".encode()).hexdigest()
    
    def get_user_patterns(self, user_id: str = None, pattern_type: str = None, 
                         limit: int = 20) -> List[Dict]:
        """Get learned patterns for user"""
        if user_id is None:
            user_id = self.current_session['user_id']
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if pattern_type:
                    cursor = conn.execute("""
                        SELECT * FROM conversation_patterns 
                        WHERE user_id = ? AND pattern_type = ?
                        ORDER BY frequency DESC, effectiveness_score DESC
                        LIMIT ?
                    """, (user_id, pattern_type, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM conversation_patterns 
                        WHERE user_id = ?
                        ORDER BY frequency DESC, effectiveness_score DESC
                        LIMIT ?
                    """, (user_id, limit))
                
                patterns = []
                for row in cursor.fetchall():
                    pattern = {
                        'id': row[0],
                        'pattern_type': row[1],
                        'pattern_data': json.loads(row[3]),
                        'frequency': row[4],
                        'effectiveness_score': row[5],
                        'last_seen': row[7]
                    }
                    patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            self.logger.error(f"Error getting user patterns: {e}")
            return []
    
    def update_user_profile(self, user_id: str):
        """Update user profile with latest interaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if profile exists
                cursor = conn.execute("SELECT interaction_count FROM user_profiles WHERE user_id = ?", (user_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing profile
                    conn.execute("""
                        UPDATE user_profiles 
                        SET interaction_count = interaction_count + 1,
                            last_interaction = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, (user_id,))
                else:
                    # Create new profile
                    conn.execute("""
                        INSERT INTO user_profiles 
                        (user_id, display_name, interaction_count)
                        VALUES (?, ?, 1)
                    """, (user_id, user_id))
                
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
    
    def get_user_profile(self, user_id: str = None) -> Dict:
        """Get user profile with caching"""
        if user_id is None:
            user_id = self.current_session['user_id']
        
        # Check cache first
        if user_id in self.user_profile_cache:
            return self.user_profile_cache[user_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    profile = {
                        'user_id': row[0],
                        'display_name': row[1],
                        'preferences': json.loads(row[2] or '{}'),
                        'conversation_style': row[3],
                        'learning_history': json.loads(row[4] or '[]'),
                        'interaction_count': row[5],
                        'first_interaction': row[6],
                        'last_interaction': row[7],
                        'profile_data': json.loads(row[8] or '{}')
                    }
                else:
                    # Create default profile
                    profile = {
                        'user_id': user_id,
                        'display_name': user_id,
                        'preferences': {},
                        'conversation_style': 'adaptive',
                        'learning_history': [],
                        'interaction_count': 0,
                        'first_interaction': datetime.now().isoformat(),
                        'last_interaction': datetime.now().isoformat(),
                        'profile_data': {}
                    }
                
                # Cache the profile
                self.user_profile_cache[user_id] = profile
                return profile
                
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return {'user_id': user_id, 'error': str(e)}
    
    def load_user_profiles(self):
        """Load user profiles into cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM user_profiles")
                
                for row in cursor.fetchall():
                    profile = {
                        'user_id': row[0],
                        'display_name': row[1],
                        'preferences': json.loads(row[2] or '{}'),
                        'conversation_style': row[3],
                        'learning_history': json.loads(row[4] or '[]'),
                        'interaction_count': row[5],
                        'first_interaction': row[6],
                        'last_interaction': row[7],
                        'profile_data': json.loads(row[8] or '{}')
                    }
                    
                    self.user_profile_cache[row[0]] = profile
                
                self.logger.info(f"Loaded {len(self.user_profile_cache)} user profiles")
                
        except Exception as e:
            self.logger.error(f"Error loading user profiles: {e}")
    
    def start_new_session(self, user_id: str = None) -> str:
        """Start a new conversation session"""
        if user_id is None:
            user_id = self.current_session['user_id']
        
        # Save current session if it has messages
        if self.current_session['message_count'] > 0:
            self.end_current_session()
        
        # Start new session
        new_session_id = self.generate_session_id()
        
        self.current_session = {
            'session_id': new_session_id,
            'user_id': user_id,
            'start_time': datetime.now(),
            'message_count': 0,
            'context_summary': "",
            'conversation_buffer': deque(maxlen=50)
        }
        
        self.metrics['session_count'] += 1
        
        self.logger.info(f"Started new session: {new_session_id}")
        return new_session_id
    
    def end_current_session(self):
        """End the current session and save summary"""
        if self.current_session['message_count'] == 0:
            return
        
        # Process any pending batches
        if self.pending_messages:
            self.process_message_batch()
        if self.pending_patterns:
            self.process_pattern_batch()
        
        # Create session summary
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get session messages
                cursor = conn.execute("""
                    SELECT content, message_type FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp
                """, (self.current_session['session_id'],))
                
                messages = cursor.fetchall()
                
                # Generate summary
                summary = self.generate_session_summary(messages)
                
                # Save session summary
                conn.execute("""
                    INSERT OR REPLACE INTO session_summaries 
                    (session_id, user_id, start_time, end_time, message_count, summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.current_session['session_id'],
                    self.current_session['user_id'],
                    self.current_session['start_time'],
                    datetime.now(),
                    self.current_session['message_count'],
                    summary
                ))
                
                self.logger.info(f"Session ended: {self.current_session['session_id']}")
                
        except Exception as e:
            self.logger.error(f"Error ending session: {e}")
    
    def generate_session_summary(self, messages: List[Tuple]) -> str:
        """Generate a summary of the session"""
        if not messages:
            return "Empty session"
        
        user_messages = [msg[0] for msg in messages if msg[1] == 'user']
        assistant_messages = [msg[0] for msg in messages if msg[1] == 'assistant']
        
        # Simple extractive summary
        key_topics = []
        if user_messages:
            # Extract key topics from user messages
            for msg in user_messages:
                words = msg.lower().split()
                important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                key_topics.extend(important_words[:3])  # Top 3 words per message
        
        # Deduplicate and limit
        key_topics = list(set(key_topics))[:10]
        
        summary = f"Session with {len(messages)} messages. "
        if key_topics:
            summary += f"Key topics: {', '.join(key_topics)}. "
        
        summary += f"User messages: {len(user_messages)}, Assistant responses: {len(assistant_messages)}"
        
        return summary
    
    def get_conversation_statistics(self) -> Dict:
        """Get comprehensive conversation statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                stats['total_messages'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                stats['unique_users'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
                stats['total_sessions'] = cursor.fetchone()[0]
                
                # Message type distribution
                cursor = conn.execute("""
                    SELECT message_type, COUNT(*) 
                    FROM conversations 
                    GROUP BY message_type
                """)
                stats['message_types'] = dict(cursor.fetchall())
                
                # Recent activity
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM conversations 
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
                stats['messages_24h'] = cursor.fetchone()[0]
                
                # Pattern statistics
                cursor = conn.execute("SELECT COUNT(*) FROM conversation_patterns")
                stats['learned_patterns'] = cursor.fetchone()[0]
                
                # Performance metrics
                stats['performance_metrics'] = self.metrics.copy()
                
                # Cache statistics
                stats['cache_sizes'] = {
                    'conversation_cache': len(self.conversation_cache),
                    'pattern_cache': len(self.pattern_cache),
                    'user_profile_cache': len(self.user_profile_cache),
                    'session_cache': len(self.session_cache)
                }
                
                # Current session info
                stats['current_session'] = {
                    'session_id': self.current_session['session_id'],
                    'user_id': self.current_session['user_id'],
                    'message_count': self.current_session['message_count'],
                    'duration_minutes': (datetime.now() - self.current_session['start_time']).total_seconds() / 60
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_old_conversations(self, days_threshold: int = 365):
        """Clean up old conversations to manage database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete old conversations with low importance
                cursor = conn.execute("""
                    DELETE FROM conversations 
                    WHERE timestamp < datetime('now', '-{} days')
                    AND importance_score < 0.3
                """.format(days_threshold))
                
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old conversations")
                    
                    # Vacuum database to reclaim space
                    conn.execute("VACUUM")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up conversations: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Process pending batches
            if self.pending_messages:
                self.process_message_batch()
            if self.pending_patterns:
                self.process_pattern_batch()
            
            # End current session
            self.end_current_session()
            
        except Exception:
            pass  # Ignore errors during cleanup


class ConversationPatternLearner:
    """Helper class for learning conversation patterns"""
    
    def __init__(self):
        self.pattern_types = {
            'question_response': self.learn_question_response_pattern,
            'topic_transition': self.learn_topic_transition_pattern,
            'user_preference': self.learn_user_preference_pattern,
            'conversation_flow': self.learn_conversation_flow_pattern
        }
    
    def learn_question_response_pattern(self, user_input: str, assistant_response: str) -> Dict:
        """Learn question-response patterns"""
        return {
            'question_type': self.classify_question_type(user_input),
            'response_style': self.classify_response_style(assistant_response),
            'user_input_length': len(user_input.split()),
            'response_length': len(assistant_response.split())
        }
    
    def learn_topic_transition_pattern(self, previous_topic: str, current_topic: str) -> Dict:
        """Learn topic transition patterns"""
        return {
            'from_topic': previous_topic,
            'to_topic': current_topic,
            'transition_type': 'smooth' if self.is_related_topic(previous_topic, current_topic) else 'abrupt'
        }
    
    def learn_user_preference_pattern(self, user_feedback: Dict) -> Dict:
        """Learn user preference patterns"""
        return {
            'feedback_type': user_feedback.get('type', 'unknown'),
            'satisfaction_score': user_feedback.get('score', 0.5),
            'preference_indicators': user_feedback.get('preferences', [])
        }
    
    def learn_conversation_flow_pattern(self, conversation_sequence: List[str]) -> Dict:
        """Learn conversation flow patterns"""
        return {
            'sequence_length': len(conversation_sequence),
            'flow_type': self.classify_flow_type(conversation_sequence),
            'engagement_level': self.calculate_engagement_level(conversation_sequence)
        }
    
    def classify_question_type(self, question: str) -> str:
        """Classify question types"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'which', 'who']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'why']):
            return 'explanatory'
        elif any(word in question_lower for word in ['can', 'could', 'would']):
            return 'capability'
        else:
            return 'general'
    
    def classify_response_style(self, response: str) -> str:
        """Classify response styles"""
        word_count = len(response.split())
        
        if word_count < 10:
            return 'concise'
        elif word_count < 50:
            return 'moderate'
        else:
            return 'detailed'
    
    def is_related_topic(self, topic1: str, topic2: str) -> bool:
        """Check if topics are related"""
        # Simple word overlap check
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())
        
        overlap = len(words1.intersection(words2))
        return overlap > 0
    
    def classify_flow_type(self, sequence: List[str]) -> str:
        """Classify conversation flow types"""
        if len(sequence) < 3:
            return 'short'
        elif len(sequence) < 10:
            return 'medium'
        else:
            return 'long'
    
    def calculate_engagement_level(self, sequence: List[str]) -> float:
        """Calculate engagement level from conversation sequence"""
        if not sequence:
            return 0.0
        
        # Simple heuristic based on response lengths and frequency
        avg_length = sum(len(msg.split()) for msg in sequence) / len(sequence)
        
        if avg_length > 20:
            return 0.9
        elif avg_length > 10:
            return 0.7
        elif avg_length > 5:
            return 0.5
        else:
            return 0.3


# Factory function
def create_optimized_conversation_memory(db_path: str = "data/conversations.db") -> OptimizedConversationMemory:
    """Create optimized conversation memory instance"""
    return OptimizedConversationMemory(db_path=db_path)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Initializing Optimized Conversation Memory...")
    start_time = time.time()
    
    memory = create_optimized_conversation_memory()
    
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f} seconds")
    
    # Test message addition
    print("\nTesting message addition...")
    
    test_messages = [
        ("Hello, how are you today?", "user"),
        ("I'm doing well, thank you for asking! How can I help you?", "assistant"),
        ("I'm learning about Python programming", "user"),
        ("That's great! Python is an excellent language to learn. What specific topics are you interested in?", "assistant"),
        ("I want to learn about machine learning", "user"),
        ("Machine learning with Python is very popular! I'd recommend starting with libraries like scikit-learn.", "assistant")
    ]
    
    for content, msg_type in test_messages:
        message_id = memory.add_message(content, msg_type, batch_mode=False)
        print(f"Added {msg_type} message {message_id}: {content[:50]}...")
    
    # Test conversation history
    print("\nTesting conversation history retrieval...")
    
    start_time = time.time()
    history = memory.get_conversation_history()
    retrieval_time = time.time() - start_time
    
    print(f"Retrieved {len(history)} messages in {retrieval_time:.3f} seconds")
    
    for i, msg in enumerate(history):
        print(f"  {i+1}. [{msg['message_type']}] {msg['content'][:60]}...")
    
    # Test pattern learning
    print("\nTesting pattern learning...")
    
    memory.learn_conversation_pattern('question_response', {
        'question_type': 'factual',
        'response_style': 'detailed',
        'topic': 'programming'
    })
    
    patterns = memory.get_user_patterns()
    print(f"Learned patterns: {len(patterns)}")
    
    for pattern in patterns:
        print(f"  Pattern: {pattern['pattern_type']} (frequency: {pattern['frequency']})")
    
    # Show statistics
    print("\nConversation Memory Statistics:")
    stats = memory.get_conversation_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # End session
    memory.end_current_session()

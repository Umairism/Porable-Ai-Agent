import faiss
import numpy as np
import sqlite3
import json
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pickle
import hashlib


class KnowledgeBase:
    """
    Local knowledge base with vector search capabilities for offline operation
    """
    
    def __init__(self, db_path: str = "knowledge/", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        
        # Initialize directories
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize embedding model (lightweight for offline use)
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Warning: Could not load embedding model {embedding_model}: {e}")
            print("Using dummy embeddings for development")
            self.embedding_model = None
            self.embedding_dim = 384  # Default dimension
        
        # Initialize FAISS index for vector search
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # SQLite database for structured data
        self.db_file = os.path.join(db_path, "knowledge.db")
        self.init_database()
        
        # In-memory cache for frequently accessed items
        self.cache = {}
        self.cache_size = 1000
        
        # Load existing knowledge
        self.load_knowledge()
    
    def init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Knowledge items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE,
                content TEXT,
                title TEXT,
                category TEXT,
                source TEXT,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Learning feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER,
                query TEXT,
                relevance_score REAL,
                user_feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
            )
        ''')
        
        # User interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                satisfaction_score REAL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.embedding_model:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        else:
            # Dummy embedding for development
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def add_knowledge(self, content: str, title: str = "", category: str = "general",
                     source: str = "user", importance: float = 1.0, metadata: Dict = None) -> int:
        """Add new knowledge item to the database"""
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Generate embedding
        embedding = self.generate_embedding(content)
        
        # Store in database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO knowledge_items 
                (content_hash, content, title, category, source, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (content_hash, content, title, category, source, importance, 
                  json.dumps(metadata) if metadata else None))
            
            knowledge_id = cursor.lastrowid
            conn.commit()
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            
            # Update cache
            self.cache[knowledge_id] = {
                'content': content,
                'title': title,
                'category': category,
                'embedding': embedding
            }
            
            # Maintain cache size
            if len(self.cache) > self.cache_size:
                # Remove least accessed items (simplified LRU)
                oldest_key = min(self.cache.keys())
                del self.cache[oldest_key]
            
            print(f"Added knowledge item: {title[:50]}... (ID: {knowledge_id})")
            return knowledge_id
            
        except sqlite3.IntegrityError:
            # Content already exists
            cursor.execute('SELECT id FROM knowledge_items WHERE content_hash = ?', (content_hash,))
            existing_id = cursor.fetchone()[0]
            print(f"Knowledge item already exists (ID: {existing_id})")
            return existing_id
        finally:
            conn.close()
    
    def search_knowledge(self, query: str, top_k: int = 5, category: str = None) -> List[Dict]:
        """Search for relevant knowledge items"""
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(top_k * 2, self.index.ntotal))
        
        # Get detailed information from database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            # Get knowledge item (idx corresponds to insertion order)
            cursor.execute('''
                SELECT id, content, title, category, source, importance, access_count, metadata
                FROM knowledge_items 
                WHERE id = (SELECT id FROM knowledge_items ORDER BY id LIMIT 1 OFFSET ?)
            ''', (int(idx),))
            
            row = cursor.fetchone()
            if row:
                knowledge_id, content, title, cat, source, importance, access_count, metadata = row
                
                # Filter by category if specified
                if category and cat != category:
                    continue
                
                # Update access count
                cursor.execute('UPDATE knowledge_items SET access_count = access_count + 1 WHERE id = ?', 
                             (knowledge_id,))
                
                results.append({
                    'id': knowledge_id,
                    'content': content,
                    'title': title,
                    'category': cat,
                    'source': source,
                    'importance': importance,
                    'access_count': access_count + 1,
                    'relevance_score': float(score),
                    'metadata': json.loads(metadata) if metadata else {}
                })
                
                if len(results) >= top_k:
                    break
        
        conn.commit()
        conn.close()
        
        # Sort by relevance score and importance
        results.sort(key=lambda x: x['relevance_score'] * x['importance'], reverse=True)
        
        return results[:top_k]
    
    def get_knowledge_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get knowledge items by category"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, content, title, category, source, importance, access_count, metadata
            FROM knowledge_items 
            WHERE category = ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
        ''', (category, limit))
        
        results = []
        for row in cursor.fetchall():
            knowledge_id, content, title, cat, source, importance, access_count, metadata = row
            results.append({
                'id': knowledge_id,
                'content': content,
                'title': title,
                'category': cat,
                'source': source,
                'importance': importance,
                'access_count': access_count,
                'metadata': json.loads(metadata) if metadata else {}
            })
        
        conn.close()
        return results
    
    def update_knowledge_importance(self, knowledge_id: int, new_importance: float):
        """Update the importance of a knowledge item based on usage patterns"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE knowledge_items 
            SET importance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_importance, knowledge_id))
        
        conn.commit()
        conn.close()
    
    def record_user_interaction(self, query: str, response: str, satisfaction_score: float = None, 
                              context: str = "general"):
        """Record user interaction for learning purposes"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions 
            (query, response, satisfaction_score, context)
            VALUES (?, ?, ?, ?)
        ''', (query, response, satisfaction_score, context))
        
        conn.commit()
        conn.close()
    
    def record_feedback(self, knowledge_id: int, query: str, relevance_score: float, 
                       user_feedback: str = ""):
        """Record feedback on knowledge relevance"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_feedback 
            (knowledge_id, query, relevance_score, user_feedback)
            VALUES (?, ?, ?, ?)
        ''', (knowledge_id, query, relevance_score, user_feedback))
        
        # Update knowledge importance based on feedback
        cursor.execute('''
            SELECT AVG(relevance_score) FROM learning_feedback WHERE knowledge_id = ?
        ''', (knowledge_id,))
        
        avg_relevance = cursor.fetchone()[0]
        if avg_relevance:
            # Adjust importance based on average relevance
            new_importance = min(2.0, max(0.1, avg_relevance))
            cursor.execute('''
                UPDATE knowledge_items 
                SET importance = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_importance, knowledge_id))
        
        conn.commit()
        conn.close()
    
    def learn_from_interactions(self):
        """Analyze user interactions to improve knowledge base"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Find frequently queried topics that might need more knowledge
        cursor.execute('''
            SELECT query, COUNT(*) as frequency, AVG(satisfaction_score) as avg_satisfaction
            FROM user_interactions 
            WHERE satisfaction_score IS NOT NULL
            GROUP BY query
            HAVING frequency > 2 AND avg_satisfaction < 0.7
            ORDER BY frequency DESC
        ''')
        
        low_satisfaction_queries = cursor.fetchall()
        
        # Find underutilized knowledge that might be outdated
        cursor.execute('''
            SELECT id, title, content, access_count, importance
            FROM knowledge_items
            WHERE access_count = 0 AND created_at < datetime('now', '-30 days')
        ''')
        
        unused_knowledge = cursor.fetchall()
        
        conn.close()
        
        # Report findings
        insights = {
            'low_satisfaction_queries': low_satisfaction_queries,
            'unused_knowledge_count': len(unused_knowledge),
            'recommendations': []
        }
        
        for query, freq, avg_sat in low_satisfaction_queries:
            insights['recommendations'].append(
                f"Consider adding more knowledge about: '{query}' (asked {freq} times, {avg_sat:.2f} satisfaction)"
            )
        
        if unused_knowledge:
            insights['recommendations'].append(
                f"Consider reviewing {len(unused_knowledge)} unused knowledge items for relevance"
            )
        
        return insights
    
    def save_knowledge(self):
        """Save FAISS index and other volatile data"""
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save cache
        cache_path = os.path.join(self.db_path, "cache.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
        
        print("Knowledge base saved successfully")
    
    def load_knowledge(self):
        """Load FAISS index and other volatile data"""
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print("FAISS index loaded successfully")
        
        # Load cache
        cache_path = os.path.join(self.db_path, "cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            print("Knowledge cache loaded successfully")
        
        # Sync database count with FAISS index
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM knowledge_items')
        db_count = cursor.fetchone()[0]
        conn.close()
        
        if db_count != self.index.ntotal:
            print(f"Warning: Database has {db_count} items but FAISS index has {self.index.ntotal}")
            if db_count > self.index.ntotal:
                print("Rebuilding FAISS index from database...")
                self.rebuild_index()
    
    def rebuild_index(self):
        """Rebuild FAISS index from database"""
        # Clear existing index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Get all knowledge items
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT id, content FROM knowledge_items ORDER BY id')
        
        embeddings = []
        for knowledge_id, content in cursor.fetchall():
            embedding = self.generate_embedding(content)
            embeddings.append(embedding)
        
        conn.close()
        
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.index.add(embeddings_array)
            print(f"Rebuilt FAISS index with {len(embeddings)} items")
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute('SELECT COUNT(*) FROM knowledge_items')
        total_items = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM user_interactions')
        total_interactions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM learning_feedback')
        total_feedback = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute('SELECT category, COUNT(*) FROM knowledge_items GROUP BY category')
        categories = dict(cursor.fetchall())
        
        # Top accessed knowledge
        cursor.execute('''
            SELECT title, access_count FROM knowledge_items 
            ORDER BY access_count DESC LIMIT 5
        ''')
        top_accessed = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_knowledge_items': total_items,
            'total_user_interactions': total_interactions,
            'total_feedback_entries': total_feedback,
            'categories': categories,
            'top_accessed_items': top_accessed,
            'faiss_index_size': self.index.ntotal,
            'cache_size': len(self.cache)
        }

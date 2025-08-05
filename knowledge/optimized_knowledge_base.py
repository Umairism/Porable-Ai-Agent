"""
Optimized Knowledge Base with Performance Improvements
Addresses bottlenecks identified in the performance analysis.
"""

import sqlite3
import faiss
import numpy as np
import json
import os
import pickle
import threading
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import hashlib
import time
from functools import lru_cache
from collections import defaultdict, deque
import logging


class OptimizedKnowledgeBase:
    """
    High-performance knowledge base with caching, batching, and async operations
    """
    
    def __init__(self, db_path: str = "data/knowledge.db", 
                 faiss_index_path: str = "data/knowledge_index.faiss",
                 embedding_dim: int = 384, cache_size: int = 10000):
        
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        
        # Initialize database with optimized settings
        self.init_database()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.knowledge_ids = []  # Maps FAISS index positions to knowledge IDs
        
        # Caching systems
        self.embedding_cache = {}
        self.search_cache = {}
        self.knowledge_cache = {}
        
        # Batch processing
        self.pending_additions = deque()
        self.batch_size = 50
        self.auto_batch_enabled = True
        
        # Threading for async operations
        self.lock = threading.RLock()
        self.background_tasks = []
        
        # Performance metrics
        self.metrics = {
            'total_searches': 0,
            'cache_hits': 0,
            'total_additions': 0,
            'batch_updates': 0,
            'average_search_time': 0.0,
            'index_size': 0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self.load_existing_data()
    
    def init_database(self):
        """Initialize optimized SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            conn.execute("PRAGMA temp_store=memory")  # In-memory temp tables
            
            # Create optimized tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance_score REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON knowledge_items(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON knowledge_items(category)")
            
            # Check if importance_score column exists before creating index
            cursor = conn.execute("PRAGMA table_info(knowledge_items)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'importance_score' in columns:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON knowledge_items(importance_score DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON knowledge_items(access_count DESC)")
            else:
                # Add missing columns for backward compatibility
                try:
                    conn.execute("ALTER TABLE knowledge_items ADD COLUMN importance_score REAL DEFAULT 1.0")
                    conn.execute("ALTER TABLE knowledge_items ADD COLUMN access_count INTEGER DEFAULT 0")
                    conn.execute("ALTER TABLE knowledge_items ADD COLUMN last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                    conn.execute("ALTER TABLE knowledge_items ADD COLUMN metadata TEXT DEFAULT '{}'")
                    
                    # Now create the indexes
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON knowledge_items(importance_score DESC)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON knowledge_items(access_count DESC)")
                    print("✅ Database schema updated for optimization")
                except sqlite3.OperationalError as e:
                    print(f"⚠️  Database schema update warning: {e}")
            
            
            # Knowledge relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER,
                    target_id INTEGER,
                    relationship_type TEXT DEFAULT 'related',
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES knowledge_items (id),
                    FOREIGN KEY (target_id) REFERENCES knowledge_items (id)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships ON knowledge_relationships(source_id, target_id)")
            
            # Knowledge embeddings table (for caching)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                    knowledge_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
                )
            """)
            
            # Knowledge usage statistics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER,
                    query_text TEXT,
                    relevance_score REAL,
                    user_feedback REAL DEFAULT NULL,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
                )
            """)
            
            conn.commit()
    
    def get_content_hash(self, content: str) -> str:
        """Generate consistent hash for content deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, content_hash: str, content: str) -> np.ndarray:
        """Get embedding with LRU caching"""
        # Check database cache first
        if content_hash in self.embedding_cache:
            return self.embedding_cache[content_hash]
        
        # Check database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM knowledge_embeddings ke "
                "JOIN knowledge_items ki ON ke.knowledge_id = ki.id "
                "WHERE ki.content_hash = ?", (content_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                embedding = pickle.loads(row[0])
                self.embedding_cache[content_hash] = embedding
                return embedding
        
        # Generate new embedding (this should use the optimized embedding system)
        from core.optimized_ai_engine import CachedEmbeddingSystem
        embedding_system = CachedEmbeddingSystem()
        embedding = embedding_system.get_embedding(content)
        
        # Cache the embedding
        self.embedding_cache[content_hash] = embedding
        
        return embedding
    
    def add_knowledge(self, content: str, category: str = "general", 
                     importance_score: float = 1.0, metadata: Dict = None,
                     batch_mode: bool = None) -> Optional[int]:
        """Add knowledge item with optional batching"""
        if batch_mode is None:
            batch_mode = self.auto_batch_enabled
        
        content_hash = self.get_content_hash(content)
        
        knowledge_item = {
            'content': content,
            'content_hash': content_hash,
            'category': category,
            'importance_score': importance_score,
            'metadata': json.dumps(metadata or {}),
            'timestamp': datetime.now()
        }
        
        if batch_mode:
            # Add to pending batch
            self.pending_additions.append(knowledge_item)
            
            # Process batch if it's full
            if len(self.pending_additions) >= self.batch_size:
                return self.process_batch_additions()
            
            return None  # Will be processed in batch
        else:
            # Process immediately
            return self._add_knowledge_immediate(knowledge_item)
    
    def _add_knowledge_immediate(self, knowledge_item: Dict) -> Optional[int]:
        """Add single knowledge item immediately"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if content already exists
                cursor = conn.execute(
                    "SELECT id FROM knowledge_items WHERE content_hash = ?",
                    (knowledge_item['content_hash'],)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update access count and importance
                    conn.execute("""
                        UPDATE knowledge_items 
                        SET access_count = access_count + 1,
                            importance_score = MAX(importance_score, ?),
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (knowledge_item['importance_score'], existing[0]))
                    
                    return existing[0]
                else:
                    # Insert new knowledge
                    cursor = conn.execute("""
                        INSERT INTO knowledge_items 
                        (content, content_hash, category, importance_score, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        knowledge_item['content'],
                        knowledge_item['content_hash'],
                        knowledge_item['category'],
                        knowledge_item['importance_score'],
                        knowledge_item['metadata']
                    ))
                    
                    knowledge_id = cursor.lastrowid
                    
                    # Generate and store embedding
                    embedding = self.get_cached_embedding(
                        knowledge_item['content_hash'], 
                        knowledge_item['content']
                    )
                    
                    # Store embedding in database
                    conn.execute("""
                        INSERT INTO knowledge_embeddings (knowledge_id, embedding)
                        VALUES (?, ?)
                    """, (knowledge_id, pickle.dumps(embedding)))
                    
                    # Add to FAISS index
                    self.index.add(embedding.reshape(1, -1))
                    self.knowledge_ids.append(knowledge_id)
                    
                    self.metrics['total_additions'] += 1
                    self.metrics['index_size'] = len(self.knowledge_ids)
                    
                    return knowledge_id
                    
        except sqlite3.IntegrityError:
            # Handle duplicate content
            return None
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            return None
    
    def process_batch_additions(self) -> List[int]:
        """Process pending knowledge additions in batch"""
        if not self.pending_additions:
            return []
        
        added_ids = []
        batch = list(self.pending_additions)
        self.pending_additions.clear()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare batch data
                batch_data = []
                batch_embeddings = []
                new_items = []
                
                for item in batch:
                    # Check for duplicates
                    cursor = conn.execute(
                        "SELECT id FROM knowledge_items WHERE content_hash = ?",
                        (item['content_hash'],)
                    )
                    existing = cursor.fetchone()
                    
                    if not existing:
                        batch_data.append((
                            item['content'],
                            item['content_hash'],
                            item['category'],
                            item['importance_score'],
                            item['metadata']
                        ))
                        
                        # Generate embedding
                        embedding = self.get_cached_embedding(
                            item['content_hash'], 
                            item['content']
                        )
                        batch_embeddings.append(embedding)
                        new_items.append(item)
                
                if batch_data:
                    # Batch insert knowledge items
                    cursor = conn.executemany("""
                        INSERT INTO knowledge_items 
                        (content, content_hash, category, importance_score, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, batch_data)
                    
                    # Get the inserted IDs
                    first_id = cursor.lastrowid
                    knowledge_ids = list(range(first_id, first_id + len(batch_data)))
                    
                    # Batch insert embeddings
                    embedding_data = [
                        (kid, pickle.dumps(emb)) 
                        for kid, emb in zip(knowledge_ids, batch_embeddings)
                    ]
                    
                    conn.executemany("""
                        INSERT INTO knowledge_embeddings (knowledge_id, embedding)
                        VALUES (?, ?)
                    """, embedding_data)
                    
                    # Batch add to FAISS index
                    if batch_embeddings:
                        embeddings_matrix = np.vstack(batch_embeddings)
                        self.index.add(embeddings_matrix)
                        self.knowledge_ids.extend(knowledge_ids)
                    
                    added_ids = knowledge_ids
                    
                    self.metrics['total_additions'] += len(batch_data)
                    self.metrics['batch_updates'] += 1
                    self.metrics['index_size'] = len(self.knowledge_ids)
                    
                    self.logger.info(f"Batch processed: {len(batch_data)} items added")
                
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            # Re-add failed items to pending queue
            self.pending_additions.extend(batch)
        
        return added_ids
    
    @lru_cache(maxsize=500)
    def search_knowledge_cached(self, query_hash: str, query: str, limit: int = 10,
                               category: str = None, min_importance: float = 0.0) -> List[Dict]:
        """Cached knowledge search"""
        return self._search_knowledge_internal(query, limit, category, min_importance)
    
    def _search_knowledge_internal(self, query: str, limit: int = 10,
                                  category: str = None, min_importance: float = 0.0) -> List[Dict]:
        """Internal search implementation"""
        try:
            # Generate query embedding
            query_hash = self.get_content_hash(query)
            query_embedding = self.get_cached_embedding(query_hash, query)
            
            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1), 
                min(limit * 3, len(self.knowledge_ids))  # Get more candidates for filtering
            )
            
            # Get candidate knowledge IDs
            candidate_ids = [self.knowledge_ids[idx] for idx in indices[0] if idx < len(self.knowledge_ids)]
            candidate_scores = scores[0]
            
            # Filter and rank results
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                # Build dynamic query based on filters
                query_parts = ["SELECT * FROM knowledge_items WHERE id IN ({})".format(
                    ','.join('?' * len(candidate_ids))
                )]
                params = candidate_ids
                
                if category:
                    query_parts.append("AND category = ?")
                    params.append(category)
                
                if min_importance > 0:
                    query_parts.append("AND importance_score >= ?")
                    params.append(min_importance)
                
                query_parts.append("ORDER BY importance_score DESC, access_count DESC")
                
                cursor = conn.execute(' '.join(query_parts), params)
                
                # Process results
                id_to_score = dict(zip(candidate_ids, candidate_scores))
                
                for row in cursor.fetchall():
                    knowledge_id = row[0]
                    similarity_score = id_to_score.get(knowledge_id, 0.0)
                    
                    result = {
                        'id': knowledge_id,
                        'content': row[1],
                        'category': row[3],
                        'importance_score': row[4],
                        'access_count': row[5],
                        'similarity_score': float(similarity_score),
                        'combined_score': float(similarity_score) * row[4],  # similarity * importance
                        'metadata': json.loads(row[8] or '{}')
                    }
                    
                    results.append(result)
                
                # Sort by combined score and limit
                results.sort(key=lambda x: x['combined_score'], reverse=True)
                results = results[:limit]
                
                # Update access statistics
                if results:
                    access_updates = [(r['id'],) for r in results]
                    conn.executemany("""
                        UPDATE knowledge_items 
                        SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, access_updates)
                
                # Record usage statistics
                usage_data = [
                    (r['id'], query, r['similarity_score']) 
                    for r in results
                ]
                conn.executemany("""
                    INSERT INTO knowledge_usage (knowledge_id, query_text, relevance_score)
                    VALUES (?, ?, ?)
                """, usage_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in knowledge search: {e}")
            return []
    
    def search_knowledge(self, query: str, limit: int = 10, category: str = None,
                        min_importance: float = 0.0) -> List[Dict]:
        """Search knowledge with caching and metrics"""
        start_time = time.time()
        
        # Generate cache key
        cache_key = f"{query}|{limit}|{category}|{min_importance}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check cache
        if cache_hash in self.search_cache:
            self.metrics['cache_hits'] += 1
            return self.search_cache[cache_hash]
        
        # Perform search
        query_hash = self.get_content_hash(query)
        results = self.search_knowledge_cached(query_hash, query, limit, category, min_importance)
        
        # Cache results
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_hash] = results
        
        # Update metrics
        search_time = time.time() - start_time
        self.metrics['total_searches'] += 1
        self.metrics['average_search_time'] = (
            (self.metrics['average_search_time'] * (self.metrics['total_searches'] - 1) + search_time) /
            self.metrics['total_searches']
        )
        
        return results
    
    async def search_knowledge_async(self, query: str, limit: int = 10, 
                                   category: str = None, min_importance: float = 0.0) -> List[Dict]:
        """Async knowledge search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.search_knowledge, query, limit, category, min_importance
        )
    
    def add_knowledge_relationship(self, source_id: int, target_id: int, 
                                  relationship_type: str = "related", 
                                  strength: float = 1.0):
        """Add relationship between knowledge items"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_relationships 
                    (source_id, target_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                """, (source_id, target_id, relationship_type, strength))
        except Exception as e:
            self.logger.error(f"Error adding relationship: {e}")
    
    def get_related_knowledge(self, knowledge_id: int, limit: int = 5) -> List[Dict]:
        """Get related knowledge items"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT ki.*, kr.relationship_type, kr.strength
                    FROM knowledge_items ki
                    JOIN knowledge_relationships kr ON ki.id = kr.target_id
                    WHERE kr.source_id = ?
                    ORDER BY kr.strength DESC, ki.importance_score DESC
                    LIMIT ?
                """, (knowledge_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'id': row[0],
                        'content': row[1],
                        'category': row[3],
                        'importance_score': row[4],
                        'relationship_type': row[-2],
                        'relationship_strength': row[-1],
                        'metadata': json.loads(row[8] or '{}')
                    }
                    results.append(result)
                
                return results
        except Exception as e:
            self.logger.error(f"Error getting related knowledge: {e}")
            return []
    
    def get_knowledge_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """Get knowledge items by category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM knowledge_items 
                    WHERE category = ?
                    ORDER BY importance_score DESC, access_count DESC
                    LIMIT ?
                """, (category, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'id': row[0],
                        'content': row[1],
                        'category': row[3],
                        'importance_score': row[4],
                        'access_count': row[5],
                        'metadata': json.loads(row[8] or '{}')
                    }
                    results.append(result)
                
                return results
        except Exception as e:
            self.logger.error(f"Error getting knowledge by category: {e}")
            return []
    
    def update_knowledge_importance(self, knowledge_id: int, new_importance: float):
        """Update importance score of knowledge item"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE knowledge_items 
                    SET importance_score = ?
                    WHERE id = ?
                """, (new_importance, knowledge_id))
        except Exception as e:
            self.logger.error(f"Error updating importance: {e}")
    
    def get_knowledge_statistics(self) -> Dict:
        """Get comprehensive knowledge base statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_items")
                total_items = cursor.fetchone()[0]
                
                # Category distribution
                cursor = conn.execute("""
                    SELECT category, COUNT(*) 
                    FROM knowledge_items 
                    GROUP BY category 
                    ORDER BY COUNT(*) DESC
                """)
                categories = dict(cursor.fetchall())
                
                # Top accessed items
                cursor = conn.execute("""
                    SELECT content, access_count 
                    FROM knowledge_items 
                    ORDER BY access_count DESC 
                    LIMIT 5
                """)
                top_accessed = cursor.fetchall()
                
                # Recent additions
                cursor = conn.execute("""
                    SELECT COUNT(*) 
                    FROM knowledge_items 
                    WHERE created_at >= datetime('now', '-24 hours')
                """)
                recent_additions = cursor.fetchone()[0]
                
                stats = {
                    'total_items': total_items,
                    'categories': categories,
                    'top_accessed': top_accessed,
                    'recent_additions_24h': recent_additions,
                    'faiss_index_size': len(self.knowledge_ids),
                    'cache_sizes': {
                        'embedding_cache': len(self.embedding_cache),
                        'search_cache': len(self.search_cache),
                        'knowledge_cache': len(self.knowledge_cache)
                    },
                    'performance_metrics': self.metrics
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def save_index(self):
        """Save FAISS index to disk"""
        try:
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            faiss.write_index(self.index, self.faiss_index_path)
            
            # Save knowledge IDs mapping
            ids_path = self.faiss_index_path.replace('.faiss', '_ids.pkl')
            with open(ids_path, 'wb') as f:
                pickle.dump(self.knowledge_ids, f)
            
            self.logger.info("FAISS index saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def load_existing_data(self):
        """Load existing FAISS index and rebuild if necessary"""
        try:
            # Try to load existing FAISS index
            if os.path.exists(self.faiss_index_path):
                self.index = faiss.read_index(self.faiss_index_path)
                
                # Load knowledge IDs mapping
                ids_path = self.faiss_index_path.replace('.faiss', '_ids.pkl')
                if os.path.exists(ids_path):
                    with open(ids_path, 'rb') as f:
                        self.knowledge_ids = pickle.load(f)
                    
                    self.logger.info(f"Loaded FAISS index with {len(self.knowledge_ids)} items")
                else:
                    self.rebuild_index()
            else:
                self.rebuild_index()
                
        except Exception as e:
            self.logger.error(f"Error loading existing data: {e}")
            self.rebuild_index()
    
    def rebuild_index(self):
        """Rebuild FAISS index from database"""
        try:
            self.logger.info("Rebuilding FAISS index...")
            
            # Reset index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.knowledge_ids = []
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all knowledge items with embeddings
                cursor = conn.execute("""
                    SELECT ki.id, ke.embedding
                    FROM knowledge_items ki
                    JOIN knowledge_embeddings ke ON ki.id = ke.knowledge_id
                    ORDER BY ki.id
                """)
                
                embeddings = []
                knowledge_ids = []
                
                for row in cursor.fetchall():
                    knowledge_id = row[0]
                    embedding = pickle.loads(row[1])
                    
                    embeddings.append(embedding)
                    knowledge_ids.append(knowledge_id)
                
                if embeddings:
                    # Add all embeddings to index
                    embeddings_matrix = np.vstack(embeddings)
                    self.index.add(embeddings_matrix)
                    self.knowledge_ids = knowledge_ids
                    
                    self.logger.info(f"Rebuilt FAISS index with {len(knowledge_ids)} items")
                    
                    # Save the rebuilt index
                    self.save_index()
                
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {e}")
    
    def cleanup_old_data(self, days_threshold: int = 90):
        """Clean up old, unused knowledge items"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete unused items older than threshold
                cursor = conn.execute("""
                    DELETE FROM knowledge_items 
                    WHERE access_count = 0 
                    AND created_at < datetime('now', '-{} days')
                    AND importance_score < 0.5
                """.format(days_threshold))
                
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old knowledge items")
                    # Rebuild index after cleanup
                    self.rebuild_index()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Process any pending additions
            if self.pending_additions:
                self.process_batch_additions()
            
            # Save index
            self.save_index()
            
        except Exception:
            pass  # Ignore errors during cleanup


# Factory function
def create_optimized_knowledge_base(db_path: str = "data/knowledge.db") -> OptimizedKnowledgeBase:
    """Create an optimized knowledge base instance"""
    return OptimizedKnowledgeBase(db_path=db_path)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Initializing Optimized Knowledge Base...")
    start_time = time.time()
    
    kb = create_optimized_knowledge_base()
    
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f} seconds")
    
    # Test knowledge addition
    print("\nTesting knowledge addition...")
    
    test_knowledge = [
        ("Python is a high-level programming language", "programming", 1.0),
        ("Machine learning uses algorithms to learn patterns", "ai", 0.9),
        ("FAISS is a library for efficient similarity search", "tools", 0.8),
        ("SQLite is a lightweight database engine", "database", 0.7),
        ("Natural language processing helps computers understand text", "ai", 0.9)
    ]
    
    for content, category, importance in test_knowledge:
        kid = kb.add_knowledge(content, category, importance)
        print(f"Added knowledge item {kid}: {content[:50]}...")
    
    # Process any pending batch additions
    kb.process_batch_additions()
    
    # Test search
    print("\nTesting knowledge search...")
    
    search_queries = [
        "What is Python?",
        "machine learning algorithms",
        "database systems",
        "natural language"
    ]
    
    for query in search_queries:
        start_time = time.time()
        results = kb.search_knowledge(query, limit=3)
        search_time = time.time() - start_time
        
        print(f"\nQuery: {query}")
        print(f"Search time: {search_time:.3f} seconds")
        print(f"Results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result['similarity_score']:.3f}] {result['content'][:60]}...")
    
    # Show statistics
    print("\nKnowledge Base Statistics:")
    stats = kb.get_knowledge_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")

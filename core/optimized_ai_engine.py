"""
Optimized AI Engine with Performance Improvements
This version addresses the major bottlenecks identified in the performance analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Deque
import numpy as np
import json
import os
from datetime import datetime
import pickle
from collections import deque, defaultdict
import asyncio
import threading
import time
from functools import lru_cache
import hashlib


class OptimizedTransformer(nn.Module):
    """
    Optimized transformer with proper tokenization and efficient architecture
    """
    
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-small", 
                 adaptation_dim: int = 128):
        super(OptimizedTransformer, self).__init__()
        
        # Use pre-trained model as base (much more efficient than training from scratch)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Add special tokens for our use case
        special_tokens = ["<USER>", "<ASSISTANT>", "<CONTEXT>", "<KNOWLEDGE>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Lightweight adaptation layers (only these are trained)
        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, adaptation_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adaptation_dim, self.base_model.config.hidden_size)
        )
        
        # Response generation head
        self.response_head = nn.Linear(
            self.base_model.config.hidden_size, 
            len(self.tokenizer)
        )
        
        # Freeze base model parameters (only train adaptation layers)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # User-specific adaptation parameters
        self.user_embeddings = nn.Embedding(100, adaptation_dim)  # Support 100 users
        self.current_user_id = 0
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                user_id: int = 0):
        # Get base model outputs
        with torch.no_grad():
            base_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = base_outputs.last_hidden_state
        
        # Apply user-specific adaptation
        user_emb = self.user_embeddings(torch.tensor([user_id]))
        adaptation = self.adaptation_layer(hidden_states)
        
        # Combine base outputs with user adaptation
        adapted_outputs = hidden_states + adaptation
        
        # Generate response logits
        logits = self.response_head(adapted_outputs)
        
        return logits
    
    def encode_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Proper tokenization using transformer tokenizer"""
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Proper detokenization"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class BatchedLearningSystem:
    """
    Efficient batch learning system to prevent catastrophic forgetting
    """
    
    def __init__(self, batch_size: int = 16, update_frequency: int = 10):
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        self.interaction_count = 0
        
        # Batch processing
        self.pending_updates = []
        
    def store_interaction(self, input_text: str, response: str, feedback_score: float,
                         context: str = "general"):
        """Store interaction for batch learning"""
        experience = {
            'input': input_text,
            'response': response,
            'feedback': feedback_score,
            'context': context,
            'timestamp': datetime.now()
        }
        
        self.experience_buffer.append(experience)
        self.interaction_count += 1
        
        # Trigger batch update if needed
        if self.interaction_count % self.update_frequency == 0:
            return True  # Signal that batch update is needed
        return False
    
    def get_training_batch(self) -> List[Dict]:
        """Get balanced batch for training"""
        if len(self.experience_buffer) < self.batch_size:
            return list(self.experience_buffer)
        
        # Sample recent and random experiences for balanced learning
        recent_samples = list(self.experience_buffer)[-self.batch_size//2:]
        
        remaining_slots = self.batch_size - len(recent_samples)
        if remaining_slots > 0:
            import random
            older_samples = random.sample(
                list(self.experience_buffer)[:-self.batch_size//2], 
                min(remaining_slots, len(self.experience_buffer) - len(recent_samples))
            )
            return recent_samples + older_samples
        
        return recent_samples


class CachedEmbeddingSystem:
    """
    Efficient embedding system with caching and async processing
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 10000):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_file = "embeddings_cache.pkl"
        
        # Load existing cache
        self.load_cache()
        
        # Async processing
        self.embedding_queue = asyncio.Queue()
        self.processing_embeddings = False
    
    def get_text_hash(self, text: str) -> str:
        """Generate consistent hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_embedding_cached(self, text_hash: str, text: str) -> np.ndarray:
        """Get embedding with LRU cache"""
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Generate new embedding
        embedding = self.model.encode(text, normalize_embeddings=True)
        
        # Store in cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text_hash] = embedding
        return embedding
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        text_hash = self.get_text_hash(text)
        return self.get_embedding_cached(text_hash, text)
    
    async def get_embedding_async(self, text: str) -> np.ndarray:
        """Get embedding asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)
    
    def load_cache(self):
        """Load embedding cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"Could not load embedding cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            print(f"Could not save embedding cache: {e}")


class OptimizedSelfLearningCore:
    """
    Optimized AI engine with performance improvements
    """
    
    def __init__(self, model_path: str = "models/", user_id: int = 0):
        self.model_path = model_path
        self.user_id = user_id
        
        # Initialize optimized components
        self.model = OptimizedTransformer()
        self.learning_system = BatchedLearningSystem()
        self.embedding_system = CachedEmbeddingSystem()
        
        # Optimizer for only trainable parameters
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Response caching
        self.response_cache = {}
        self.cache_max_size = 1000
        
        # Performance tracking
        self.performance_metrics = {
            'total_interactions': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'learning_events': 0
        }
        
        # Template-based responses for common patterns
        self.response_templates = self.load_response_templates()
        
        # Load existing model
        self.load_model()
    
    def load_response_templates(self) -> Dict:
        """Load response templates for common patterns"""
        return {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Welcome! I'm here to assist you."
            ],
            'farewell': [
                "Goodbye! Feel free to ask me anything anytime.",
                "See you later! I'll remember our conversation.",
                "Take care! I'm always here when you need help."
            ],
            'uncertainty': [
                "I'm not entirely sure about that. Could you provide more context?",
                "That's an interesting question. Let me think about it...",
                "I'd like to learn more about this topic. Can you tell me more?"
            ],
            'knowledge_request': [
                "Based on what I know about {topic}...",
                "Here's what I can tell you about {topic}...",
                "Let me share some information about {topic}..."
            ]
        }
    
    def get_response_hash(self, input_text: str, context: str = "") -> str:
        """Generate hash for response caching"""
        combined = f"{input_text}|{context}|{self.user_id}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    @lru_cache(maxsize=500)
    def generate_response_cached(self, input_hash: str, input_text: str, 
                                context: str = "general") -> str:
        """Generate response with caching"""
        return self._generate_response_internal(input_text, context)
    
    def _generate_response_internal(self, input_text: str, context: str) -> str:
        """Internal response generation"""
        # Check for template-based responses first (fastest)
        template_response = self.try_template_response(input_text, context)
        if template_response:
            return template_response
        
        # Use model for complex responses
        self.model.eval()
        
        with torch.no_grad():
            # Encode input
            encoded = self.model.encode_text(f"<USER> {input_text} <ASSISTANT>")
            
            # Generate response
            logits = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                user_id=self.user_id
            )
            
            # Simple greedy decoding (can be improved with beam search)
            predicted_ids = torch.argmax(logits[:, -1, :], dim=-1)
            
            # Decode response
            response = self.model.decode_tokens(predicted_ids)
            
            # Ensure response quality
            if len(response.strip()) < 3:
                response = "I understand. Could you tell me more about what you're looking for?"
        
        return response
    
    def try_template_response(self, input_text: str, context: str) -> Optional[str]:
        """Try to use template-based response for common patterns"""
        input_lower = input_text.lower().strip()
        
        # Greeting detection
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            import random
            return random.choice(self.response_templates['greeting'])
        
        # Farewell detection
        if any(word in input_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            import random
            return random.choice(self.response_templates['farewell'])
        
        # Question detection
        if input_text.strip().endswith('?') and len(input_text.split()) > 3:
            # This is a question, use model
            return None
        
        return None
    
    async def generate_response_async(self, input_text: str, context: str = "general") -> str:
        """Generate response asynchronously"""
        start_time = time.time()
        
        # Check cache first
        response_hash = self.get_response_hash(input_text, context)
        if response_hash in self.response_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.response_cache[response_hash]
        
        # Generate new response
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.generate_response_cached, response_hash, input_text, context
        )
        
        # Cache response
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[response_hash] = response
        
        # Update metrics
        response_time = time.time() - start_time
        self.performance_metrics['total_interactions'] += 1
        self.performance_metrics['average_response_time'] = (
            (self.performance_metrics['average_response_time'] * 
             (self.performance_metrics['total_interactions'] - 1) + response_time) /
            self.performance_metrics['total_interactions']
        )
        
        return response
    
    def generate_response(self, input_text: str, context: str = "general") -> str:
        """Synchronous response generation"""
        response_hash = self.get_response_hash(input_text, context)
        return self.generate_response_cached(response_hash, input_text, context)
    
    def learn_from_feedback(self, input_text: str, expected_output: str, 
                           feedback_score: float, context: str = "general"):
        """Store feedback for batch learning"""
        # Store in batch learning system
        should_update = self.learning_system.store_interaction(
            input_text, expected_output, feedback_score, context
        )
        
        if should_update:
            # Perform batch update
            self.perform_batch_update()
    
    def perform_batch_update(self):
        """Perform efficient batch learning update"""
        batch_data = self.learning_system.get_training_batch()
        
        if not batch_data:
            return
        
        self.model.train()
        
        # Prepare batch tensors
        inputs = []
        targets = []
        
        for item in batch_data:
            # Encode input and target
            input_encoded = self.model.encode_text(f"<USER> {item['input']} <ASSISTANT>")
            target_encoded = self.model.encode_text(item['response'])
            
            inputs.append(input_encoded)
            targets.append(target_encoded)
        
        # Batch forward pass
        total_loss = 0
        for inp, tgt in zip(inputs, targets):
            logits = self.model(
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                user_id=self.user_id
            )
            
            # Calculate loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt['input_ids'].view(-1)
            )
            total_loss += loss
        
        # Average loss and backward pass
        avg_loss = total_loss / len(inputs)
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        self.performance_metrics['learning_events'] += 1
        print(f"Batch learning completed: Loss={avg_loss.item():.4f}, Batch size={len(inputs)}")
    
    def save_model(self):
        """Save optimized model and caches"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save only trainable parameters
        trainable_state = {
            name: param for name, param in self.model.state_dict().items()
            if any(name.startswith(layer) for layer in ['adaptation_layer', 'response_head', 'user_embeddings'])
        }
        
        torch.save(trainable_state, os.path.join(self.model_path, 'adaptation_weights.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(self.model_path, 'optimizer.pth'))
        
        # Save response cache
        with open(os.path.join(self.model_path, 'response_cache.pkl'), 'wb') as f:
            pickle.dump(self.response_cache, f)
        
        # Save performance metrics
        with open(os.path.join(self.model_path, 'performance_metrics.json'), 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save embedding cache
        self.embedding_system.save_cache()
        
        print("Optimized model saved successfully")
    
    def load_model(self):
        """Load optimized model and caches"""
        adaptation_file = os.path.join(self.model_path, 'adaptation_weights.pth')
        
        if os.path.exists(adaptation_file):
            # Load adaptation weights
            adaptation_state = torch.load(adaptation_file, map_location='cpu')
            
            # Load only the adaptation layers
            model_state = self.model.state_dict()
            model_state.update(adaptation_state)
            self.model.load_state_dict(model_state)
            
            # Load optimizer state
            optimizer_file = os.path.join(self.model_path, 'optimizer.pth')
            if os.path.exists(optimizer_file):
                self.optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))
            
            # Load response cache
            cache_file = os.path.join(self.model_path, 'response_cache.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.response_cache = pickle.load(f)
            
            # Load performance metrics
            metrics_file = os.path.join(self.model_path, 'performance_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
            
            print("Optimized model loaded successfully")
        else:
            print("No existing optimized model found, starting fresh")
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        stats = self.performance_metrics.copy()
        
        # Calculate additional metrics
        if stats['total_interactions'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_interactions']
        else:
            stats['cache_hit_rate'] = 0.0
        
        stats['response_cache_size'] = len(self.response_cache)
        stats['embedding_cache_size'] = len(self.embedding_system.cache)
        stats['experience_buffer_size'] = len(self.learning_system.experience_buffer)
        
        return stats


# Factory function for easy instantiation
def create_optimized_ai_engine(model_path: str = "models/", user_id: int = 0) -> OptimizedSelfLearningCore:
    """Create an optimized AI engine instance"""
    return OptimizedSelfLearningCore(model_path=model_path, user_id=user_id)


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    print("Initializing Optimized AI Engine...")
    start_time = time.time()
    
    ai = create_optimized_ai_engine()
    
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f} seconds")
    
    # Test response generation
    print("\nTesting response generation...")
    
    test_inputs = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Can you help me learn Python?",
        "Thank you and goodbye!"
    ]
    
    for i, input_text in enumerate(test_inputs):
        start_time = time.time()
        response = ai.generate_response(input_text)
        response_time = time.time() - start_time
        
        print(f"\nTest {i+1}:")
        print(f"Input: {input_text}")
        print(f"Response: {response}")
        print(f"Time: {response_time:.3f} seconds")
    
    # Show performance stats
    print("\nPerformance Statistics:")
    stats = ai.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save the model
    ai.save_model()

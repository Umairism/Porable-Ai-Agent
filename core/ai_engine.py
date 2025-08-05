import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import os
from datetime import datetime
import pickle


class AdaptiveTransformer(nn.Module):
    """
    Lightweight transformer model with adaptive learning capabilities
    """
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 4, max_seq_len: int = 512):
        super(AdaptiveTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # Learning rate adaptation
        self.learning_rates = {}
        self.adaptation_factor = 0.01
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        embeddings = self.embedding(input_ids) + self.pos_encoding(pos_ids)
        embeddings = self.dropout(embeddings)
        
        # Transformer
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            output = self.transformer(embeddings, src_key_padding_mask=~attention_mask)
        else:
            output = self.transformer(embeddings)
        
        # Output projection
        logits = self.output_projection(output)
        return logits
    
    def adapt_learning_rate(self, loss: float, context: str):
        """Adapt learning rate based on performance and context"""
        if context not in self.learning_rates:
            self.learning_rates[context] = 1e-4
        
        current_lr = self.learning_rates[context]
        
        # Increase LR if loss is high, decrease if low
        if loss > 2.0:
            new_lr = current_lr * (1 + self.adaptation_factor)
        elif loss < 0.5:
            new_lr = current_lr * (1 - self.adaptation_factor * 0.5)
        else:
            new_lr = current_lr
        
        self.learning_rates[context] = max(1e-6, min(1e-2, new_lr))
        return self.learning_rates[context]


class SelfLearningCore:
    """
    Core AI engine with self-learning capabilities
    """
    
    def __init__(self, model_path: str = "models/", vocab_size: int = 10000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        
        # Initialize model
        self.model = AdaptiveTransformer(vocab_size=vocab_size)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning history
        self.learning_history = []
        self.conversation_context = []
        self.feedback_buffer = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_interactions': 0,
            'successful_responses': 0,
            'learning_events': 0,
            'adaptation_count': 0
        }
        
        # Load existing model if available
        self.load_model()
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Simple tokenization and encoding"""
        # This is a simplified tokenizer - in practice, you'd use a proper tokenizer
        tokens = text.lower().split()[:self.model.max_seq_len-2]
        
        # Simple vocabulary mapping (would be replaced with proper tokenizer)
        token_ids = [hash(token) % (self.vocab_size - 2) + 2 for token in tokens]
        token_ids = [1] + token_ids + [2]  # Add start and end tokens
        
        # Pad sequence
        while len(token_ids) < self.model.max_seq_len:
            token_ids.append(0)
        
        return torch.tensor(token_ids[:self.model.max_seq_len]).unsqueeze(0)
    
    def decode_ids(self, token_ids: torch.Tensor) -> str:
        """Simple decoding with basic template responses"""
        # Template-based responses for better user experience
        templates = [
            "I understand what you're asking about. Let me help you with that.",
            "That's an interesting question. Based on what I know, here's my response.",
            "I can help you with that. Let me provide you with some information.",
            "Thank you for your question. Here's what I think about that topic.",
            "I'm learning from our conversation. Let me give you a helpful response.",
            "That's a good point. I'd be happy to share my thoughts on that.",
            "I appreciate your question. Here's how I can assist you.",
            "Based on our conversation, I think I can help you with that."
        ]
        
        # Use hash of input to consistently return same response for same input
        import random
        random.seed(hash(str(token_ids.tolist())) % 1000)
        return random.choice(templates)
    
    def generate_response(self, input_text: str, context: str = "general") -> str:
        """Generate response and learn from interaction"""
        self.model.eval()
        
        # Encode input
        input_ids = self.encode_text(input_text)
        
        with torch.no_grad():
            # Generate response
            logits = self.model(input_ids)
            
            # Simple response generation (would be more sophisticated)
            response_ids = torch.argmax(logits, dim=-1)
            response = self.decode_ids(response_ids)
        
        # Store interaction for learning
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'response': response,
            'context': context
        }
        self.conversation_context.append(interaction)
        
        # Update metrics
        self.performance_metrics['total_interactions'] += 1
        
        return response
    
    def learn_from_feedback(self, input_text: str, expected_output: str, 
                           feedback_score: float, context: str = "general"):
        """Learn from user feedback"""
        self.model.train()
        
        # Encode input and expected output
        input_ids = self.encode_text(input_text)
        target_ids = self.encode_text(expected_output)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Calculate loss (simplified)
        loss = self.criterion(logits.view(-1, self.vocab_size), target_ids.view(-1))
        
        # Adapt learning rate based on feedback
        adapted_lr = self.model.adapt_learning_rate(loss.item(), context)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adapted_lr
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store learning event
        learning_event = {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'expected': expected_output,
            'loss': loss.item(),
            'learning_rate': adapted_lr,
            'context': context,
            'feedback_score': feedback_score
        }
        self.learning_history.append(learning_event)
        
        # Update metrics
        self.performance_metrics['learning_events'] += 1
        self.performance_metrics['adaptation_count'] += 1
        
        if feedback_score > 0.7:  # Consider successful if feedback score > 0.7
            self.performance_metrics['successful_responses'] += 1
        
        print(f"Learning event: Loss={loss.item():.4f}, LR={adapted_lr:.6f}")
    
    def continuous_learning(self):
        """Periodic self-improvement from accumulated interactions"""
        if len(self.conversation_context) < 10:
            return
        
        self.model.train()
        
        # Sample recent interactions for learning
        recent_interactions = self.conversation_context[-10:]
        
        for interaction in recent_interactions:
            # Self-supervised learning from conversation patterns
            input_ids = self.encode_text(interaction['input'])
            
            # Generate current response
            with torch.no_grad():
                current_logits = self.model(input_ids)
            
            # Train to be more consistent with learned patterns
            logits = self.model(input_ids)
            
            # Use self-consistency as target (simplified approach)
            loss = torch.mean((logits - current_logits.detach()) ** 2)
            
            if loss.item() > 0.1:  # Only learn if there's significant inconsistency
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        print(f"Continuous learning completed on {len(recent_interactions)} interactions")
    
    def save_model(self):
        """Save model and learning state"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'model.pth'))
        
        # Save learning history and metrics
        with open(os.path.join(self.model_path, 'learning_history.json'), 'w') as f:
            json.dump(self.learning_history, f, indent=2)
        
        with open(os.path.join(self.model_path, 'metrics.json'), 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save conversation context
        with open(os.path.join(self.model_path, 'conversation_context.pkl'), 'wb') as f:
            pickle.dump(self.conversation_context, f)
        
        print("Model and learning state saved successfully")
    
    def load_model(self):
        """Load model and learning state"""
        model_file = os.path.join(self.model_path, 'model.pth')
        
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
            print("Model loaded successfully")
            
            # Load learning history
            history_file = os.path.join(self.model_path, 'learning_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.learning_history = json.load(f)
            
            # Load metrics
            metrics_file = os.path.join(self.model_path, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
            
            # Load conversation context
            context_file = os.path.join(self.model_path, 'conversation_context.pkl')
            if os.path.exists(context_file):
                with open(context_file, 'rb') as f:
                    self.conversation_context = pickle.load(f)
        else:
            print("No existing model found, starting fresh")
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.performance_metrics.copy()
        
        if stats['total_interactions'] > 0:
            stats['success_rate'] = stats['successful_responses'] / stats['total_interactions']
            stats['learning_rate'] = stats['learning_events'] / stats['total_interactions']
        else:
            stats['success_rate'] = 0.0
            stats['learning_rate'] = 0.0
        
        stats['conversation_length'] = len(self.conversation_context)
        stats['learning_events_count'] = len(self.learning_history)
        
        return stats

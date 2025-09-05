"""
Optimized Portable AI Agent - Main Entry Point
Uses the performance-optimized components to provide fast, efficient AI interactions.
"""

import asyncio
import argparse
import sys
import time
import os
from typing import Dict, List, Optional
import logging

# Import optimized components
from core.optimized_ai_engine import OptimizedSelfLearningCore
from knowledge.optimized_knowledge_base import OptimizedKnowledgeBase
from memory.optimized_conversation_memory import OptimizedConversationMemory

# Import original interfaces (these should also be optimized in a real implementation)
from interface.cli_interface import PortableAIInterface
from interface.web_interface import WebInterface

# Import utilities
from utils.setup_check import check_dependencies
from utils.config_manager import ConfigManager


class OptimizedPortableAI:
    """
    Optimized Portable AI Agent with high-performance components
    """
    
    def __init__(self, config_path: str = "config.json", user_id: str = "default_user"):
        self.config_path = config_path
        self.user_id = user_id
        
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'total_interactions': 0,
            'average_response_time': 0.0,
            'cache_hit_rates': {},
            'component_stats': {}
        }
        
        # Initialize components
        self.ai_engine = None
        self.knowledge_base = None
        self.conversation_memory = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.initialize_components()
    
    def setup_logging(self):
        """Setup optimized logging"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/portable_ai.log', mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_components(self):
        """Initialize all AI components with performance tracking"""
        start_time = time.time()
        
        self.logger.info("Initializing Optimized Portable AI Agent...")
        
        try:
            # Initialize AI Engine
            ai_start = time.time()
            self.ai_engine = OptimizedSelfLearningCore(
                model_path=self.config.get('model_path', 'models/'),
                user_id=hash(self.user_id) % 100  # Map to user embedding index
            )
            ai_time = time.time() - ai_start
            self.logger.info(f"AI Engine initialized in {ai_time:.2f}s")
            
            # Initialize Knowledge Base
            kb_start = time.time()
            self.knowledge_base = OptimizedKnowledgeBase(
                db_path=self.config.get('knowledge_db_path', 'data/knowledge.db'),
                embedding_dim=self.config.get('embedding_dim', 384),
                cache_size=self.config.get('knowledge_cache_size', 10000)
            )
            kb_time = time.time() - kb_start
            self.logger.info(f"Knowledge Base initialized in {kb_time:.2f}s")
            
            # Initialize Conversation Memory
            memory_start = time.time()
            self.conversation_memory = OptimizedConversationMemory(
                db_path=self.config.get('conversation_db_path', 'data/conversations.db'),
                cache_size=self.config.get('memory_cache_size', 5000),
                batch_size=self.config.get('memory_batch_size', 25)
            )
            memory_time = time.time() - memory_start
            self.logger.info(f"Conversation Memory initialized in {memory_time:.2f}s")
            
            # Set user context
            self.conversation_memory.current_session['user_id'] = self.user_id
            
            total_time = time.time() - start_time
            self.performance_metrics['initialization_time'] = total_time
            
            self.logger.info(f"All components initialized successfully in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def process_input_async(self, user_input: str, context: str = "general") -> Dict:
        """
        Process user input asynchronously with performance optimization
        """
        start_time = time.time()
        
        try:
            # Add user message to memory (batched)
            self.conversation_memory.add_message(
                content=user_input,
                message_type="user",
                context_tags=[context],
                batch_mode=True
            )
            
            # Search for relevant knowledge asynchronously
            knowledge_task = asyncio.create_task(
                self.knowledge_base.search_knowledge_async(
                    query=user_input,
                    limit=self.config.get('knowledge_search_limit', 5),
                    category=context if context != "general" else None
                )
            )
            
            # Get conversation history asynchronously
            history_task = asyncio.create_task(
                self.conversation_memory.get_conversation_history_async(
                    user_id=self.user_id,
                    limit=self.config.get('context_history_limit', 10)
                )
            )
            
            # Wait for knowledge and history
            knowledge_results, conversation_history = await asyncio.gather(
                knowledge_task, history_task
            )
            
            # Build enhanced context
            enhanced_context = self.build_enhanced_context(
                user_input, knowledge_results, conversation_history, context
            )
            
            # Generate response asynchronously
            response = await self.ai_engine.generate_response_async(
                input_text=user_input,
                context=enhanced_context
            )
            
            # Add assistant response to memory (batched)
            response_time_ms = int((time.time() - start_time) * 1000)
            self.conversation_memory.add_message(
                content=response,
                message_type="assistant",
                context_tags=[context],
                response_time_ms=response_time_ms,
                batch_mode=True
            )
            
            # Learn from the interaction
            self.learn_from_interaction(user_input, response, knowledge_results, context)
            
            # Update performance metrics
            total_time = time.time() - start_time
            self.update_performance_metrics(total_time)
            
            return {
                'response': response,
                'context_used': enhanced_context,
                'knowledge_sources': len(knowledge_results),
                'response_time': total_time,
                'processing_stats': {
                    'knowledge_search_results': len(knowledge_results),
                    'history_messages': len(conversation_history),
                    'response_time_ms': response_time_ms
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return {
                'response': "I encountered an error processing your request. Please try again.",
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def process_input(self, user_input: str, context: str = "general") -> Dict:
        """
        Synchronous wrapper for input processing
        """
        return asyncio.run(self.process_input_async(user_input, context))
    
    def build_enhanced_context(self, user_input: str, knowledge_results: List[Dict], 
                              conversation_history: List[Dict], context: str) -> str:
        """
        Build enhanced context from knowledge and conversation history
        """
        context_parts = [f"Context: {context}"]
        
        # Add relevant knowledge
        if knowledge_results:
            context_parts.append("Relevant Knowledge:")
            for i, knowledge in enumerate(knowledge_results[:3]):  # Top 3 results
                context_parts.append(
                    f"  {i+1}. [{knowledge['similarity_score']:.3f}] {knowledge['content'][:200]}..."
                )
        
        # Add recent conversation context
        if conversation_history:
            context_parts.append("Recent Conversation:")
            for msg in conversation_history[-5:]:  # Last 5 messages
                msg_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                context_parts.append(f"  {msg['message_type']}: {msg_preview}")
        
        return "\n".join(context_parts)
    
    def learn_from_interaction(self, user_input: str, response: str, 
                              knowledge_results: List[Dict], context: str):
        """
        Learn from the interaction to improve future responses
        """
        try:
            # Simple feedback scoring based on knowledge relevance
            feedback_score = 0.5  # Default neutral score
            
            if knowledge_results:
                # Higher score if we found relevant knowledge
                avg_similarity = sum(k['similarity_score'] for k in knowledge_results) / len(knowledge_results)
                feedback_score = min(0.9, 0.5 + avg_similarity * 0.4)
            
            # Learn from feedback (will be batched)
            self.ai_engine.learn_from_feedback(
                input_text=user_input,
                expected_output=response,
                feedback_score=feedback_score,
                context=context
            )
            
            # Learn conversation patterns
            self.conversation_memory.learn_conversation_pattern(
                pattern_type='question_response',
                pattern_data={
                    'input_length': len(user_input.split()),
                    'response_length': len(response.split()),
                    'context': context,
                    'knowledge_used': len(knowledge_results),
                    'response_quality': feedback_score
                },
                effectiveness_score=feedback_score
            )
            
            # Add new knowledge if the interaction was educational
            if len(user_input.split()) > 5 and feedback_score > 0.7:
                combined_knowledge = f"Q: {user_input} A: {response}"
                self.knowledge_base.add_knowledge(
                    content=combined_knowledge,
                    category=context,
                    importance_score=feedback_score,
                    metadata={
                        'source': 'conversation',
                        'user_id': self.user_id,
                        'interaction_quality': feedback_score
                    },
                    batch_mode=True
                )
            
        except Exception as e:
            self.logger.error(f"Error in learning from interaction: {e}")
    
    def update_performance_metrics(self, response_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_interactions'] += 1
        
        # Update average response time
        total = self.performance_metrics['total_interactions']
        current_avg = self.performance_metrics['average_response_time']
        
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total - 1) + response_time) / total
        )
        
        # Collect component stats
        if hasattr(self.ai_engine, 'get_performance_stats'):
            self.performance_metrics['component_stats']['ai_engine'] = \
                self.ai_engine.get_performance_stats()
        
        if hasattr(self.knowledge_base, 'get_knowledge_statistics'):
            kb_stats = self.knowledge_base.get_knowledge_statistics()
            self.performance_metrics['component_stats']['knowledge_base'] = kb_stats
        
        if hasattr(self.conversation_memory, 'get_conversation_statistics'):
            memory_stats = self.conversation_memory.get_conversation_statistics()
            self.performance_metrics['component_stats']['conversation_memory'] = memory_stats
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        report = {
            'system_metrics': self.performance_metrics.copy(),
            'component_details': {},
            'optimization_recommendations': []
        }
        
        # AI Engine Performance
        if self.ai_engine:
            ai_stats = self.ai_engine.get_performance_stats()
            report['component_details']['ai_engine'] = ai_stats
            
            # Recommendations for AI Engine
            if ai_stats.get('cache_hit_rate', 0) < 0.3:
                report['optimization_recommendations'].append(
                    "AI Engine: Consider increasing response cache size for better performance"
                )
            
            if ai_stats.get('average_response_time', 0) > 2.0:
                report['optimization_recommendations'].append(
                    "AI Engine: Response time is high, consider model optimization"
                )
        
        # Knowledge Base Performance
        if self.knowledge_base:
            kb_stats = self.knowledge_base.get_knowledge_statistics()
            report['component_details']['knowledge_base'] = kb_stats
            
            # Recommendations for Knowledge Base
            perf_metrics = kb_stats.get('performance_metrics', {})
            if perf_metrics.get('cache_hit_rate', 0) < 0.4:
                report['optimization_recommendations'].append(
                    "Knowledge Base: Low cache hit rate, consider increasing cache size"
                )
            
            if perf_metrics.get('average_search_time', 0) > 0.5:
                report['optimization_recommendations'].append(
                    "Knowledge Base: Search time is high, consider index optimization"
                )
        
        # Memory Performance
        if self.conversation_memory:
            memory_stats = self.conversation_memory.get_conversation_statistics()
            report['component_details']['conversation_memory'] = memory_stats
            
            # Recommendations for Memory
            perf_metrics = memory_stats.get('performance_metrics', {})
            if perf_metrics.get('average_write_time', 0) > 0.1:
                report['optimization_recommendations'].append(
                    "Conversation Memory: Write time is high, consider increasing batch size"
                )
        
        return report
    
    def add_knowledge_from_file(self, file_path: str, category: str = "imported") -> int:
        """
        Add knowledge from a text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks if the file is large
            chunks = self.split_text_into_chunks(content)
            added_count = 0
            
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only add substantial chunks
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=chunk.strip(),
                        category=category,
                        importance_score=0.8,
                        metadata={
                            'source': 'file',
                            'file_path': file_path,
                            'import_time': time.time()
                        },
                        batch_mode=True
                    )
                    if knowledge_id:
                        added_count += 1
            
            # Process any pending batches
            self.knowledge_base.process_batch_additions()
            
            self.logger.info(f"Added {added_count} knowledge items from {file_path}")
            return added_count
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge from file: {e}")
            return 0
    
    def split_text_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into manageable chunks for knowledge storage
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def save_all_data(self):
        """
        Save all component data
        """
        try:
            # Save AI model
            if self.ai_engine:
                self.ai_engine.save_model()
            
            # Process any pending batches
            if self.knowledge_base and hasattr(self.knowledge_base, 'process_batch_additions'):
                self.knowledge_base.process_batch_additions()
                self.knowledge_base.save_index()
            
            if self.conversation_memory:
                if hasattr(self.conversation_memory, 'process_message_batch'):
                    self.conversation_memory.process_message_batch()
                if hasattr(self.conversation_memory, 'process_pattern_batch'):
                    self.conversation_memory.process_pattern_batch()
                self.conversation_memory.end_current_session()
            
            # Save configuration
            self.config_manager.save_config(self.config)
            
            self.logger.info("All data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def interactive_mode(self):
        """
        Run in interactive command-line mode
        """
        print("🤖 Optimized Portable AI Agent - Interactive Mode")
        print("=" * 50)
        print("Type 'exit' to quit, 'help' for commands, 'stats' for performance stats")
        print()
        
        while True:
            try:
                user_input = input(f"[{self.user_id}] You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_performance_stats()
                    continue
                elif user_input.lower().startswith('load '):
                    file_path = user_input[5:].strip()
                    if os.path.exists(file_path):
                        count = self.add_knowledge_from_file(file_path)
                        print(f"✅ Added {count} knowledge items from {file_path}")
                    else:
                        print(f"❌ File not found: {file_path}")
                    continue
                elif user_input.lower() == 'save':
                    self.save_all_data()
                    print("✅ All data saved")
                    continue
                elif not user_input:
                    continue
                
                # Process the input
                print("🤔 Processing...", end="", flush=True)
                result = self.process_input(user_input)
                print("\r" + " " * 15 + "\r", end="")  # Clear processing message
                
                # Display response
                print(f"🤖 AI: {result['response']}")
                
                # Show performance info
                if result.get('response_time'):
                    print(f"   ⏱️  Response time: {result['response_time']:.3f}s")
                    if result.get('knowledge_sources', 0) > 0:
                        print(f"   📚 Knowledge sources used: {result['knowledge_sources']}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                self.logger.error(f"Error in interactive mode: {e}")
    
    def show_help(self):
        """Show help information"""
        print("""
🔧 Available Commands:
  help          - Show this help message
  stats         - Show performance statistics
  load <file>   - Load knowledge from a text file
  save          - Save all data to disk
  exit/quit/bye - Exit the application

💡 Tips:
  - Ask questions naturally
  - The AI learns from your interactions
  - Knowledge is automatically saved from conversations
  - Use specific contexts for better responses
        """)
    
    def show_performance_stats(self):
        """Show current performance statistics"""
        report = self.get_performance_report()
        
        print("\n📊 Performance Statistics")
        print("=" * 40)
        
        # System metrics
        system = report['system_metrics']
        print(f"Initialization time: {system['initialization_time']:.2f}s")
        print(f"Total interactions: {system['total_interactions']}")
        print(f"Average response time: {system['average_response_time']:.3f}s")
        
        # Component details
        if 'ai_engine' in report['component_details']:
            ai_stats = report['component_details']['ai_engine']
            print(f"\n🧠 AI Engine:")
            print(f"  Cache hit rate: {ai_stats.get('cache_hit_rate', 0):.1%}")
            print(f"  Learning events: {ai_stats.get('learning_events', 0)}")
        
        if 'knowledge_base' in report['component_details']:
            kb_stats = report['component_details']['knowledge_base']
            print(f"\n📚 Knowledge Base:")
            print(f"  Total items: {kb_stats.get('total_items', 0)}")
            print(f"  Cache sizes: {kb_stats.get('cache_sizes', {})}")
        
        if 'conversation_memory' in report['component_details']:
            mem_stats = report['component_details']['conversation_memory']
            print(f"\n💭 Conversation Memory:")
            print(f"  Total messages: {mem_stats.get('total_messages', 0)}")
            print(f"  Learned patterns: {mem_stats.get('learned_patterns', 0)}")
        
        # Recommendations
        if report['optimization_recommendations']:
            print(f"\n🔧 Optimization Recommendations:")
            for rec in report['optimization_recommendations']:
                print(f"  • {rec}")
        
        print()
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.save_all_data()
        except Exception:
            pass  # Ignore errors during cleanup


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Optimized Portable AI Agent")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--user-id", default="default_user", help="User ID for personalization")
    parser.add_argument("--mode", choices=["interactive", "cli", "web"], default="interactive",
                       help="Interface mode")
    parser.add_argument("--input", help="Direct input for single query mode")
    parser.add_argument("--load-knowledge", help="Load knowledge from text file")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Missing dependencies. Please run setup.py install")
        sys.exit(1)
    
    try:
        # Initialize AI agent
        print("🚀 Starting Optimized Portable AI Agent...")
        ai_agent = OptimizedPortableAI(config_path=args.config, user_id=args.user_id)
        
        # Load knowledge file if specified
        if args.load_knowledge:
            if os.path.exists(args.load_knowledge):
                count = ai_agent.add_knowledge_from_file(args.load_knowledge)
                print(f"✅ Loaded {count} knowledge items from {args.load_knowledge}")
            else:
                print(f"❌ Knowledge file not found: {args.load_knowledge}")
                sys.exit(1)
        
        # Run benchmark if requested
        if args.benchmark:
            run_benchmark(ai_agent)
            return
        
        # Handle single input mode
        if args.input:
            result = ai_agent.process_input(args.input)
            print(f"Response: {result['response']}")
            print(f"Response time: {result['response_time']:.3f}s")
            return
        
        # Choose interface mode
        if args.mode == "interactive":
            ai_agent.interactive_mode()
        elif args.mode == "cli":
            cli = PortableAIInterface()
            # Pass the optimized agent to the CLI interface
            cli.ai_engine = ai_agent.ai_engine
            cli.knowledge_base = ai_agent.knowledge_base
            cli.memory = ai_agent.conversation_memory
            cli.cmdloop()
        elif args.mode == "web":
            web = WebInterface(ai_agent)
            web.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def run_benchmark(ai_agent: OptimizedPortableAI):
    """Run performance benchmark"""
    print("\n🏃 Running Performance Benchmark...")
    print("=" * 50)
    
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain Python programming",
        "What are neural networks?",
        "How do I learn data science?",
        "What is the difference between AI and ML?",
        "Explain deep learning concepts",
        "How do chatbots work?",
        "What is natural language processing?",
        "How do recommendation systems work?"
    ]
    
    total_time = 0
    total_queries = len(test_queries)
    
    print(f"Running {total_queries} test queries...")
    
    for i, query in enumerate(test_queries):
        print(f"Query {i+1}/{total_queries}: {query[:50]}...")
        
        start_time = time.time()
        result = ai_agent.process_input(query)
        query_time = time.time() - start_time
        
        total_time += query_time
        print(f"  Response time: {query_time:.3f}s")
        
        # Brief delay between queries
        time.sleep(0.1)
    
    avg_time = total_time / total_queries
    
    print(f"\n📊 Benchmark Results:")
    print(f"  Total queries: {total_queries}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average response time: {avg_time:.3f}s")
    print(f"  Queries per second: {total_queries/total_time:.2f}")
    
    # Get detailed performance report
    report = ai_agent.get_performance_report()
    
    if report['optimization_recommendations']:
        print(f"\n🔧 Optimization Recommendations:")
        for rec in report['optimization_recommendations']:
            print(f"  • {rec}")
    
    print(f"\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()

import cmd
import sys
import os
import time
import threading
from typing import Dict, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_engine import SelfLearningCore
from knowledge.knowledge_base import KnowledgeBase
from memory.conversation_memory import ConversationMemory


class PortableAIInterface(cmd.Cmd):
    """
    Command-line interface for the Portable AI Agent
    """
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║                    Portable AI Agent                        ║
║              Offline • Self-Learning • Private              ║
╚══════════════════════════════════════════════════════════════╝

Welcome to your personal AI assistant! 
Type 'help' for available commands or just start chatting.
Type 'exit' or 'quit' to leave.
    """
    
    prompt = "🤖 > "
    
    def __init__(self):
        super().__init__()
        self.ai_engine = None
        self.knowledge_base = None
        self.memory = None
        self.conversation_active = False
        self.auto_save_thread = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize AI components"""
        print("Initializing Portable AI Agent...")
        
        try:
            # Initialize core components
            self.ai_engine = SelfLearningCore()
            self.knowledge_base = KnowledgeBase()
            self.memory = ConversationMemory()
            
            # Start new conversation session
            self.memory.start_session({"interface": "cli", "version": "1.0"})
            
            # Start auto-save thread
            self.start_auto_save()
            
            print("✅ AI Agent initialized successfully!")
            print("💡 The agent learns from every interaction to improve over time.")
            print("🔒 All data stays on your device - completely private and offline.")
            print()
            
        except Exception as e:
            print(f"❌ Error initializing AI Agent: {e}")
            print("Please check your setup and try again.")
            sys.exit(1)
    
    def start_auto_save(self):
        """Start background auto-save thread"""
        def auto_save_worker():
            while self.conversation_active:
                time.sleep(300)  # Save every 5 minutes
                if self.conversation_active:
                    try:
                        self.ai_engine.save_model()
                        self.knowledge_base.save_knowledge()
                        print("\n💾 Auto-saved agent state")
                        print(self.prompt, end="", flush=True)
                    except Exception as e:
                        print(f"\n⚠️  Auto-save error: {e}")
                        print(self.prompt, end="", flush=True)
        
        self.conversation_active = True
        self.auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_thread.start()
    
    def default(self, line):
        """Handle normal conversation input"""
        if not line.strip():
            return
        
        user_input = line.strip()
        
        # Add user message to memory
        self.memory.add_message("user", user_input, importance=1.0)
        
        # Get relevant context from memory and knowledge
        context = self.memory.get_relevant_context(user_input, max_messages=5)
        relevant_knowledge = self.knowledge_base.search_knowledge(user_input, top_k=3)
        
        # Generate response
        print("🤔 Thinking...", end="", flush=True)
        
        # Determine context for AI engine
        ai_context = "general"
        if relevant_knowledge:
            ai_context = relevant_knowledge[0].get('category', 'general')
        
        try:
            response = self.ai_engine.generate_response(user_input, context=ai_context)
            
            # Enhance response with knowledge if relevant
            if relevant_knowledge and relevant_knowledge[0]['relevance_score'] > 0.7:
                knowledge_info = relevant_knowledge[0]['content'][:200]
                enhanced_response = f"{response}\n\n💡 Related info: {knowledge_info}..."
            else:
                enhanced_response = response
            
            print(f"\r🤖 {enhanced_response}")
            
            # Add AI response to memory
            self.memory.add_message("assistant", enhanced_response, 
                                  topic=ai_context, importance=1.0)
            
            # Record interaction for learning
            self.knowledge_base.record_user_interaction(user_input, enhanced_response, context=ai_context)
            
            # Trigger continuous learning periodically
            if self.ai_engine.performance_metrics['total_interactions'] % 10 == 0:
                print("🧠 Learning from recent interactions...")
                self.ai_engine.continuous_learning()
            
        except Exception as e:
            print(f"\r❌ Error generating response: {e}")
            print("Please try rephrasing your question.")
    
    def do_learn(self, line):
        """
        Teach the AI with feedback: learn <input> | <expected_output> [score]
        Example: learn What is Python? | Python is a programming language 0.9
        """
        try:
            parts = line.split('|')
            if len(parts) < 2:
                print("Usage: learn <input> | <expected_output> [score]")
                return
            
            input_text = parts[0].strip()
            expected_output = parts[1].strip()
            
            # Parse optional score
            score = 1.0
            if len(parts) > 2:
                try:
                    score = float(parts[2].strip())
                except ValueError:
                    score = 1.0
            
            print(f"📚 Teaching: '{input_text}' -> '{expected_output}' (score: {score})")
            
            # Train the AI engine
            self.ai_engine.learn_from_feedback(input_text, expected_output, score)
            
            # Add to knowledge base
            self.knowledge_base.add_knowledge(
                content=f"Q: {input_text}\nA: {expected_output}",
                title=f"User teaching: {input_text[:50]}",
                category="user_taught",
                importance=score
            )
            
            print("✅ Learning completed! The AI will remember this for future conversations.")
            
        except Exception as e:
            print(f"❌ Error during learning: {e}")
    
    def do_knowledge(self, line):
        """
        Add knowledge to the AI: knowledge <content> [title] [category]
        Example: knowledge Python is great for data science | Python Info | programming
        """
        try:
            parts = line.split('|')
            content = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else f"Knowledge: {content[:50]}"
            category = parts[2].strip() if len(parts) > 2 else "general"
            
            knowledge_id = self.knowledge_base.add_knowledge(
                content=content,
                title=title,
                category=category,
                source="user"
            )
            
            print(f"✅ Added knowledge item (ID: {knowledge_id})")
            
        except Exception as e:
            print(f"❌ Error adding knowledge: {e}")
    
    def do_search(self, line):
        """Search the knowledge base: search <query>"""
        if not line.strip():
            print("Usage: search <query>")
            return
        
        try:
            results = self.knowledge_base.search_knowledge(line.strip(), top_k=5)
            
            if not results:
                print("No relevant knowledge found.")
                return
            
            print(f"\n🔍 Found {len(results)} relevant items:")
            print("-" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   Category: {result['category']}")
                print(f"   Relevance: {result['relevance_score']:.3f}")
                print(f"   Content: {result['content'][:150]}...")
                if i < len(results):
                    print()
            
        except Exception as e:
            print(f"❌ Error searching knowledge: {e}")
    
    def do_stats(self, line):
        """Show AI agent statistics"""
        try:
            print("\n📊 Portable AI Agent Statistics")
            print("=" * 50)
            
            # AI Engine stats
            ai_stats = self.ai_engine.get_performance_stats()
            print(f"🤖 AI Engine:")
            print(f"   Total interactions: {ai_stats['total_interactions']}")
            print(f"   Learning events: {ai_stats['learning_events']}")
            print(f"   Success rate: {ai_stats['success_rate']:.2%}")
            print(f"   Adaptation count: {ai_stats['adaptation_count']}")
            
            # Knowledge base stats
            kb_stats = self.knowledge_base.get_statistics()
            print(f"\n📚 Knowledge Base:")
            print(f"   Total items: {kb_stats['total_knowledge_items']}")
            print(f"   Categories: {len(kb_stats['categories'])}")
            print(f"   User interactions: {kb_stats['total_user_interactions']}")
            print(f"   Feedback entries: {kb_stats['total_feedback_entries']}")
            
            # Memory stats
            memory_stats = self.memory.get_conversation_statistics()
            print(f"\n💭 Memory System:")
            print(f"   Total messages: {memory_stats['total_messages']}")
            print(f"   Sessions: {memory_stats['total_sessions']}")
            print(f"   Recent messages (7d): {memory_stats['recent_messages_7d']}")
            print(f"   Current context size: {memory_stats['current_context_size']}")
            print(f"   User profile entries: {memory_stats['user_profile_entries']}")
            
            print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
    def do_profile(self, line):
        """Manage user profile: profile set <key> <value> OR profile show"""
        parts = line.split()
        
        if not parts:
            print("Usage: profile set <key> <value> OR profile show")
            return
        
        try:
            if parts[0] == "set" and len(parts) >= 3:
                key = parts[1]
                value = " ".join(parts[2:])
                self.memory.update_user_profile(key, value)
                print(f"✅ Set profile: {key} = {value}")
                
            elif parts[0] == "show":
                print("\n👤 User Profile:")
                print("-" * 30)
                
                for key, data in self.memory.user_profile.items():
                    print(f"{key}: {data['value']} (confidence: {data['confidence']:.2f})")
                
                if not self.memory.user_profile:
                    print("No profile data stored yet.")
                    print("Use 'profile set <key> <value>' to add information.")
            
            else:
                print("Usage: profile set <key> <value> OR profile show")
                
        except Exception as e:
            print(f"❌ Error managing profile: {e}")
    
    def do_save(self, line):
        """Save all AI agent data"""
        try:
            print("💾 Saving AI agent state...")
            self.ai_engine.save_model()
            self.knowledge_base.save_knowledge()
            # Memory is auto-saved to database
            print("✅ All data saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving: {e}")
    
    def do_insights(self, line):
        """Show learning insights and recommendations"""
        try:
            print("\n🔍 Learning Insights")
            print("=" * 40)
            
            # Get knowledge base insights
            insights = self.knowledge_base.learn_from_interactions()
            
            if insights['low_satisfaction_queries']:
                print("📉 Areas needing improvement:")
                for query, freq, avg_sat in insights['low_satisfaction_queries']:
                    print(f"   • '{query}' - Asked {freq} times, {avg_sat:.1f}/5.0 satisfaction")
            
            if insights['recommendations']:
                print("\n💡 Recommendations:")
                for rec in insights['recommendations']:
                    print(f"   • {rec}")
            
            if not insights['low_satisfaction_queries'] and not insights['recommendations']:
                print("🎉 Great job! No specific improvement areas identified.")
                print("Your AI agent is performing well based on current data.")
            
            print("\n📊 General Tips:")
            print("   • Use 'learn' command to teach specific responses")
            print("   • Add relevant knowledge with 'knowledge' command") 
            print("   • Provide feedback to improve future responses")
            
        except Exception as e:
            print(f"❌ Error generating insights: {e}")
    
    def do_clear(self, line):
        """Clear the screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print(self.intro)
    
    def do_help(self, line):
        """Show help information"""
        if line:
            super().do_help(line)
        else:
            print("\n🆘 Available Commands:")
            print("-" * 40)
            print("💬 Just type normally to chat with the AI")
            print("📚 learn <input> | <output> [score] - Teach the AI")
            print("🧠 knowledge <content> | [title] | [category] - Add knowledge")
            print("🔍 search <query> - Search knowledge base")
            print("📊 stats - Show agent statistics")
            print("👤 profile set/show - Manage user profile")
            print("💾 save - Save all data")
            print("🔍 insights - Show learning insights")
            print("🧹 clear - Clear screen")
            print("❓ help - Show this help")
            print("🚪 exit/quit - Exit the agent")
            print()
            print("💡 Tips:")
            print("   • The AI learns from every conversation")
            print("   • Add knowledge to improve responses")
            print("   • Use 'learn' to correct mistakes")
            print("   • Check 'stats' to monitor progress")
    
    def do_exit(self, line):
        """Exit the AI agent"""
        return self.do_quit(line)
    
    def do_quit(self, line):
        """Exit the AI agent"""
        print("\n💾 Saving final state...")
        
        try:
            # Stop auto-save
            self.conversation_active = False
            
            # Final save
            self.ai_engine.save_model()
            self.knowledge_base.save_knowledge()
            
            # Create memory summary
            self.memory.create_memory_summary("session")
            
            print("✅ All data saved successfully!")
            print("\n👋 Thank you for using Portable AI Agent!")
            print("Your AI has learned from this session and will remember for next time.")
            print("Stay curious! 🚀")
            
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
            print("Some data may not have been saved.")
        
        return True
    
    def cmdloop(self, intro=None):
        """Override cmdloop to handle KeyboardInterrupt gracefully"""
        try:
            super().cmdloop(intro)
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            self.do_quit("")


def main():
    """Main entry point for CLI interface"""
    try:
        interface = PortableAIInterface()
        interface.cmdloop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

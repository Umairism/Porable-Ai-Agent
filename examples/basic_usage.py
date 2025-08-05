#!/usr/bin/env python3
"""
Example: Basic Usage of Portable AI Agent
This script demonstrates how to use the core components programmatically
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ai_engine import SelfLearningCore
from knowledge.knowledge_base import KnowledgeBase
from memory.conversation_memory import ConversationMemory

def main():
    print("🤖 Portable AI Agent - Basic Usage Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing AI components...")
    ai_engine = SelfLearningCore()
    knowledge_base = KnowledgeBase()
    memory = ConversationMemory()
    
    print("✅ Components initialized successfully!")
    
    # Start a conversation session
    memory.start_session({"example": "basic_usage", "user": "demo"})
    
    # Example 1: Basic conversation
    print("\n📝 Example 1: Basic Conversation")
    print("-" * 30)
    
    user_input = "Hello, can you help me learn about Python?"
    print(f"User: {user_input}")
    
    # Add to memory
    memory.add_message("user", user_input, topic="programming", importance=1.0)
    
    # Get relevant context
    context = memory.get_relevant_context(user_input)
    
    # Search knowledge base
    relevant_knowledge = knowledge_base.search_knowledge(user_input, top_k=3)
    
    # Generate response
    response = ai_engine.generate_response(user_input, context="programming")
    print(f"AI: {response}")
    
    # Add AI response to memory
    memory.add_message("assistant", response, topic="programming", importance=1.0)
    
    # Example 2: Teaching the AI
    print("\n📚 Example 2: Teaching the AI")
    print("-" * 30)
    
    teaching_input = "What is machine learning?"
    expected_response = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
    
    print(f"Teaching: '{teaching_input}' -> '{expected_response}'")
    
    # Teach the AI
    ai_engine.learn_from_feedback(teaching_input, expected_response, feedback_score=0.9)
    
    # Add to knowledge base
    knowledge_id = knowledge_base.add_knowledge(
        content=f"Q: {teaching_input}\nA: {expected_response}",
        title="Machine Learning Definition",
        category="AI_concepts",
        source="user_teaching"
    )
    
    print(f"✅ Teaching completed! Knowledge ID: {knowledge_id}")
    
    # Example 3: Adding structured knowledge
    print("\n🧠 Example 3: Adding Knowledge")
    print("-" * 30)
    
    knowledge_items = [
        {
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
            "title": "Python Programming Language",
            "category": "programming"
        },
        {
            "content": "NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.",
            "title": "NumPy Library",
            "category": "python_libraries"
        },
        {
            "content": "Pandas is a data analysis and manipulation library for Python. It provides data structures like DataFrame for handling structured data efficiently.",
            "title": "Pandas Library", 
            "category": "python_libraries"
        }
    ]
    
    for item in knowledge_items:
        kb_id = knowledge_base.add_knowledge(
            content=item["content"],
            title=item["title"],
            category=item["category"],
            source="example_data"
        )
        print(f"Added: {item['title']} (ID: {kb_id})")
    
    # Example 4: Searching knowledge
    print("\n🔍 Example 4: Knowledge Search")
    print("-" * 30)
    
    search_queries = ["Python programming", "data analysis", "machine learning"]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = knowledge_base.search_knowledge(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (relevance: {result['relevance_score']:.3f})")
            print(f"     {result['content'][:100]}...")
    
    # Example 5: Memory and context
    print("\n💭 Example 5: Memory and Context")
    print("-" * 30)
    
    # Simulate a conversation
    conversation = [
        ("user", "I'm learning Python for data science"),
        ("assistant", "That's great! Python is excellent for data science with libraries like NumPy and Pandas."),
        ("user", "What should I learn first?"),
        ("assistant", "Start with Python basics, then learn NumPy for numerical computing, followed by Pandas for data manipulation."),
        ("user", "Can you recommend some practice projects?")
    ]
    
    for role, message in conversation:
        memory.add_message(role, message, topic="data_science", importance=1.0)
        print(f"{role.title()}: {message}")
    
    # Get context for the last message
    last_query = conversation[-1][1]
    context = memory.get_relevant_context(last_query, max_messages=3)
    
    print(f"\nRelevant context for '{last_query}':")
    for ctx in context:
        print(f"  - {ctx['role']}: {ctx['content'][:60]}...")
    
    # Example 6: Performance statistics
    print("\n📊 Example 6: Performance Statistics")
    print("-" * 30)
    
    # AI Engine stats
    ai_stats = ai_engine.get_performance_stats()
    print("AI Engine Statistics:")
    for key, value in ai_stats.items():
        print(f"  {key}: {value}")
    
    # Knowledge Base stats
    kb_stats = knowledge_base.get_statistics()
    print(f"\nKnowledge Base Statistics:")
    print(f"  Total items: {kb_stats['total_knowledge_items']}")
    print(f"  Categories: {list(kb_stats['categories'].keys())}")
    print(f"  User interactions: {kb_stats['total_user_interactions']}")
    
    # Memory stats
    memory_stats = memory.get_conversation_statistics()
    print(f"\nMemory Statistics:")
    print(f"  Total messages: {memory_stats['total_messages']}")
    print(f"  Current context size: {memory_stats['current_context_size']}")
    print(f"  User profile entries: {memory_stats['user_profile_entries']}")
    
    # Example 7: User profile
    print("\n👤 Example 7: User Profile")
    print("-" * 30)
    
    # Set user preferences
    profile_data = {
        "name": "Demo User",
        "interests": "Python, Data Science, AI",
        "experience_level": "Intermediate",
        "preferred_learning_style": "Examples and practice"
    }
    
    for key, value in profile_data.items():
        memory.update_user_profile(key, value, category="preferences")
        print(f"Set profile: {key} = {value}")
    
    # Example 8: Continuous learning
    print("\n🧠 Example 8: Continuous Learning")
    print("-" * 30)
    
    print("Triggering continuous learning...")
    ai_engine.continuous_learning()
    
    # Get learning insights
    insights = knowledge_base.learn_from_interactions()
    print("Learning insights:")
    for recommendation in insights.get('recommendations', []):
        print(f"  💡 {recommendation}")
    
    # Save all data
    print("\n💾 Saving Data")
    print("-" * 30)
    
    ai_engine.save_model()
    knowledge_base.save_knowledge()
    # Memory is auto-saved to database
    
    print("✅ All data saved successfully!")
    
    print("\n🎉 Example completed!")
    print("Your AI agent has learned from this session and will remember for next time.")

if __name__ == "__main__":
    main()

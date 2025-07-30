#!/usr/bin/env python3
"""
RedisAI + LangGraph Agent Demo

A simple demonstration of integrating RedisAI with LangGraph agents.
"""

import os
import sys
from dotenv import load_dotenv
from redis_ai_client import RedisAIClient
from agent import RedisAILangGraphAgent

def check_redis_connection():
    """Check if Redis is running and accessible"""
    try:
        redis_client = RedisAIClient()
        redis_client.redis_client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please make sure Redis is running. You can start it with:")
        print("docker run -d --name redis-ai -p 6379:6379 redislabs/redisai:latest")
        return False

def main():
    """Main demo function"""
    print("🚀 RedisAI + LangGraph Agent Demo")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Check Redis connection
    if not check_redis_connection():
        return
    
    # Initialize RedisAI client
    print("🔧 Initializing RedisAI client...")
    redis_client = RedisAIClient()
    
    # Initialize LangGraph agent
    print("🤖 Initializing LangGraph agent...")
    agent = RedisAILangGraphAgent(redis_client, openai_api_key)
    
    print("\n✅ Setup complete! Starting interactive chat...")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    # Interactive chat loop
    session_id = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process message through agent
            print("🤔 Processing...")
            result = agent.process_message(user_input, session_id)
            
            # Update session ID for conversation continuity
            session_id = result["session_id"]
            
            # Display RAG context used
            print(f"📚 RAG Context: {result['rag_context']}")
            
            # Display response
            print(f"🤖 Assistant: {result['response']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

def demo_workflow():
    """Demonstrate the workflow with predefined messages"""
    print("🎬 Running demo workflow...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Initialize components
    redis_client = RedisAIClient()
    agent = RedisAILangGraphAgent(redis_client, openai_api_key)
    
    # Demo messages
    demo_messages = [
        "How does AI.MODELRUN differ from AI.DAGRUN?",
        "What's the simplest way to run a model using RedisAI",
        "Can you help me with a technical question?",
        "I love this demo, it's amazing!"
    ]
    
    session_id = None
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n--- Demo Message {i} ---")
        print(f"👤 User: {message}")
        
        result = agent.process_message(message, session_id)
        session_id = result["session_id"]
        
        print(f"🤖 Assistant: {result['response']}")
    
    print("\n✅ Demo workflow completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_workflow()
    else:
        main() 
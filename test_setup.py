#!/usr/bin/env python3
"""
Test script to verify RedisAI + LangGraph setup
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import redis
        print("✅ redis imported successfully")
    except ImportError as e:
        print(f"❌ redis import failed: {e}")
        return False
    
    try:
        import langgraph
        print("✅ langgraph imported successfully")
    except ImportError as e:
        print(f"❌ langgraph import failed: {e}")
        return False
    
    try:
        import langchain
        print("✅ langchain imported successfully")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    return True

def test_redis_connection():
    """Test Redis connection"""
    print("\n🔍 Testing Redis connection...")
    
    try:
        from redis_ai_client import RedisAIClient
        redis_client = RedisAIClient()
        redis_client.redis_client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation"""
    print("\n🔍 Testing agent creation...")
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        from redis_ai_client import RedisAIClient
        from agent import RedisAILangGraphAgent
        
        redis_client = RedisAIClient()
        agent = RedisAILangGraphAgent(redis_client, openai_api_key)
        print("✅ Agent created successfully")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("\n🔍 Testing sentiment analysis...")
    
    try:
        from redis_ai_client import RedisAIClient
        
        redis_client = RedisAIClient()
        
        test_messages = [
            "I love this!",
            "This is terrible",
            "Hello world"
        ]
        
        for message in test_messages:
            sentiment = redis_client.simple_sentiment_analysis(message)
            print(f"  '{message}' -> {sentiment}")
        
        print("✅ Sentiment analysis working")
        return True
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RedisAI + LangGraph Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_redis_connection,
        test_agent_creation,
        test_sentiment_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nYou can now run:")
        print("  python main.py          # Interactive chat")
        print("  python main.py --demo   # Demo workflow")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
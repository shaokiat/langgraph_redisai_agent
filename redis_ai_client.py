import redis
import numpy as np
from typing import List, Dict, Any
import json

class RedisAIClient:
    """Wrapper for RedisAI operations"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.ai_client = redis.Redis(host=host, port=port, db=db)
    
    def store_conversation(self, session_id: str, message: str, response: str, sentiment: str):
        """Store conversation data in Redis"""
        conversation_data = {
            'message': message,
            'response': response,
            'sentiment': sentiment,
            'timestamp': str(np.datetime64('now'))
        }
        
        # Store in Redis as JSON
        self.redis_client.lpush(f"conversation:{session_id}", json.dumps(conversation_data))
        self.redis_client.expire(f"conversation:{session_id}", 3600)  # Expire in 1 hour
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve conversation history from Redis"""
        try:
            history = self.redis_client.lrange(f"conversation:{session_id}", 0, limit - 1)
            return [json.loads(item.decode('utf-8')) for item in history]
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return [] 
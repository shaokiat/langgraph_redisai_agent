import redis
import numpy as np
from typing import List, Dict, Any
import json

class RedisAIClient:
    """Wrapper for RedisAI operations"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.ai_client = redis.Redis(host=host, port=port, db=db)
    
    def load_model(self, model_name: str, model_path: str, backend: str = 'TORCH'):
        """Load a model into RedisAI"""
        try:
            # Load model from file
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            # Store model in RedisAI
            self.ai_client.execute_command('AI.MODELSTORE', model_name, backend, 'CPU', model_data)
            print(f"Model {model_name} loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def run_inference(self, model_name: str, inputs: List[str], output_keys: List[str] = None) -> Dict[str, Any]:
        """Run inference using a loaded model"""
        try:
            # Prepare input tensors
            input_tensors = []
            for i, input_text in enumerate(inputs):
                # Convert text to tensor (simple encoding for demo)
                tensor_data = np.array([ord(c) for c in input_text], dtype=np.float32)
                input_tensors.append(f"input_{i}")
                self.ai_client.execute_command('AI.TENSORSET', f"input_{i}", 'FLOAT', 1, len(tensor_data), 'BLOB', tensor_data.tobytes())
            
            # Set output keys if not provided
            if output_keys is None:
                output_keys = [f"output_{i}" for i in range(len(inputs))]
            
            # Run model inference
            self.ai_client.execute_command('AI.MODELEXECUTE', model_name, *input_tensors, *output_keys)
            
            # Get results
            results = {}
            for key in output_keys:
                tensor_info = self.ai_client.execute_command('AI.TENSORGET', key, 'META')
                tensor_data = self.ai_client.execute_command('AI.TENSORGET', key, 'BLOB')
                results[key] = {
                    'shape': tensor_info[1],
                    'data': np.frombuffer(tensor_data, dtype=np.float32)
                }
            
            return results
        except Exception as e:
            print(f"Error running inference: {e}")
            return {}
    
    def simple_sentiment_analysis(self, text: str) -> str:
        """Simple sentiment analysis using basic text processing"""
        # This is a simplified sentiment analysis for demo purposes
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'horrible']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
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
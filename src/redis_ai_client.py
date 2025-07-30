import redis
import numpy as np
import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RedisAIClient:
    """Wrapper for RedisAI operations"""

    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.ai_client = redis.Redis(host=host, port=port, db=db)

    def store_conversation(self, session_id: str, message: str, response: str):
        """Store conversation data in Redis"""
        conversation_data = {
            'message': message,
            'response': response,
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

    def store_document_with_embedding(self, doc_id: str, text: str, embedding: list, index_name: str = "rag_docs"):
        """Store a document and its embedding in Redis for vector search (RAG)."""
        # Store as a Redis hash with vector field
        pipe = self.redis_client.pipeline()
        pipe.hset(f"{index_name}:{doc_id}", mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes()
        })
        pipe.execute()

    def create_vector_index(self, index_name: str = "rag_docs", dim: int = 1536):
        """Create a Redis vector index for RAG if it doesn't exist."""
        try:
            self.redis_client.execute_command(
                "FT.CREATE", index_name, "ON", "HASH", "PREFIX", "1", f"{index_name}:",
                "SCHEMA", "text", "TEXT", "embedding", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", dim, "DISTANCE_METRIC", "COSINE"
            )
        except Exception as e:
            if "Index already exists" in str(e):
                pass  # Already created
            else:
                print(f"Error creating vector index: {e}")
    
    def embed_text(self, text: str) -> list:
        """Get embedding for text using OpenAI API."""
        resp = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return resp.data[0].embedding

    def query_similar_documents(self, query: str, k: int = 3, index_name: str = "rag_docs") -> list:
        """Query Redis for top-k similar documents using vector search."""
        embedding = self.embed_text(query)
        query_vec = np.array(embedding, dtype=np.float32).tobytes()
        # Use FT.SEARCH with vector similarity
        try:
            result = self.redis_client.execute_command(
                "FT.SEARCH", index_name,
                f"*=>[KNN {k} @embedding $vec as score]",
                "PARAMS", "2", "vec", query_vec,
                "SORTBY", "score", "ASC",
                "RETURN", "2", "text", "score",
                "DIALECT", "2",
                "LIMIT", "0", str(k))
            docs = []
            for i in range(1, len(result), 2):
                doc = result[i+1]
                docs.append({"score": float(doc[1].decode()), "text": doc[3].decode()})
            return docs
        except Exception as e:
            print(f"Error querying similar documents: {e}")
            return [] 
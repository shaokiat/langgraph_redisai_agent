# RedisAI + LangGraph Agent RAG Demo

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) agent using:

- 🧠 LangGraph for structured LLM reasoning and tool use

- 🗃️ Redis Vector Database for semantic search over documentation

- 📄 Redis OSS and RedisAI GitHub documentation as the knowledge base

The result is a powerful and interactive chatbot agent that can answer technical questions based on Redis docs.

## Features

- **Redis Vector Database**: Uses Redis VectorDB for storing of Redis documentation
- **LangGraph Agent**: Implements a simple conversational agent using LangGraph
- **Conversation Flow**: Shows how to build conversational workflows

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Redis with RedisAI**:

   **Option A: Using Docker Compose (Recommended)**

   ```bash
   docker-compose up -d
   ```

   **Option B: Using Docker directly**

   ```bash
   docker run -d --name redis-ai -p 6379:6379 redislabs/redisai:latest
   ```

   **Option C: Using RedisAI locally**
   If you have RedisAI installed locally, just start your Redis server.

3. **Environment Variables**:
   Copy the example environment file and add your OpenAI API key:

   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Initialize the VectorDB with Redis Documentation**:
   Official Redis Inference Optimization GitHub Repository:
   https://github.com/RedisAI/redis-inference-optimization

   ```bash
   # In root directory
   git clone https://github.com/RedisAI/redis-inference-optimization.git

   # Change directory to src/ folder
   python ingest_redis_doc.py
   ```

5. **Run the Demo**:

   ```bash
   # Change directory to src/ folder
   # Interactive chat mode
   python main.py

   # Demo workflow mode
   python main.py --demo
   ```

## Project Structure

- `main.py`: Main demo application
- `agent.py`: LangGraph agent implementation
- `redis_ai_client.py`: RedisAI client wrapper
- `test_setup.py`: Setup verification script
- `docker-compose.yml`: Docker setup for RedisAI
- `env.example`: Environment variables template

## Usage

The demo creates a simple conversational agent that:

1. Uses Redis Vector Database to store Redis Github documentations and retrieve relevant results
2. Processes user input through a LangGraph workflow
3. Provides responses based relevant vector results and conversation context

Run the demo and start chatting with the agent!

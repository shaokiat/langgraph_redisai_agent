# RedisAI + LangGraph Agent Demo

A simple demonstration project showing how to integrate RedisAI with LangGraph agents for AI-powered applications.

## Features

- **RedisAI Integration**: Uses RedisAI for model serving and inference
- **LangGraph Agent**: Implements a simple conversational agent using LangGraph
- **Model Management**: Demonstrates loading and serving models through RedisAI
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

4. **Test the Setup**:

   ```bash
   python test_setup.py
   ```

5. **Run the Demo**:

   ```bash
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

1. Uses RedisAI to serve a sentiment analysis model
2. Processes user input through a LangGraph workflow
3. Provides responses based on sentiment analysis and conversation context

Run the demo and start chatting with the agent!

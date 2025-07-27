from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import uuid
from redis_ai_client import RedisAIClient

# Define the state structure
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    user_input: str
    sentiment: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    response: str

class RedisAILangGraphAgent:
    """LangGraph agent that integrates with RedisAI"""
    
    def __init__(self, redis_client: RedisAIClient, openai_api_key: str):
        self.redis_client = redis_client
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_sentiment", self._analyze_sentiment)
        workflow.add_node("get_context", self._get_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("store_conversation", self._store_conversation)
        
        # Define the flow
        workflow.set_entry_point("analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "get_context")
        workflow.add_edge("get_context", "generate_response")
        workflow.add_edge("generate_response", "store_conversation")
        workflow.add_edge("store_conversation", END)
        
        return workflow.compile()
    
    def _analyze_sentiment(self, state: AgentState) -> AgentState:
        """Analyze sentiment of user input using RedisAI"""
        user_input = state["user_input"]
        
        # Use RedisAI for sentiment analysis
        sentiment = self.redis_client.simple_sentiment_analysis(user_input)
        
        print(f"Sentiment analysis: {sentiment}")
        
        return {
            **state,
            "sentiment": sentiment
        }
    
    def _get_context(self, state: AgentState) -> AgentState:
        """Get conversation context from Redis"""
        session_id = state["session_id"]
        
        # Get conversation history from Redis
        conversation_history = self.redis_client.get_conversation_history(session_id, limit=5)
        
        return {
            **state,
            "conversation_history": conversation_history
        }
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate response using LLM based on sentiment and context"""
        user_input = state["user_input"]
        sentiment = state["sentiment"]
        conversation_history = state["conversation_history"]
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "Previous conversation:\n"
            for entry in reversed(conversation_history[:3]):  # Last 3 exchanges
                context += f"User: {entry['message']}\n"
                context += f"Assistant: {entry['response']}\n"
        
        # Create prompt based on sentiment
        if sentiment == "positive":
            system_prompt = "You are a helpful and enthusiastic assistant. The user seems to be in a positive mood, so respond warmly and encouragingly."
        elif sentiment == "negative":
            system_prompt = "You are a helpful and empathetic assistant. The user seems to be in a negative mood, so respond with understanding and support."
        else:
            system_prompt = "You are a helpful and professional assistant. Respond clearly and informatively."
        
        # Prepare messages for LLM
        messages = [
            HumanMessage(content=f"{system_prompt}\n\n{context}\n\nUser: {user_input}")
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        return {
            **state,
            "response": response.content
        }
    
    def _store_conversation(self, state: AgentState) -> AgentState:
        """Store conversation in Redis"""
        session_id = state["session_id"]
        user_input = state["user_input"]
        response = state["response"]
        sentiment = state["sentiment"]
        
        # Store in Redis
        self.redis_client.store_conversation(session_id, user_input, response, sentiment)
        
        return state
    
    def process_message(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user message through the workflow"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            user_input=user_input,
            sentiment="",
            session_id=session_id,
            conversation_history=[],
            response=""
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "response": final_state["response"],
            "sentiment": final_state["sentiment"],
            "session_id": session_id
        } 
import os
from typing import List, Annotated
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Native SDK imports
from pinecone import Pinecone
import cohere

# ==========================================
# 1. Initialization & Config
# ==========================================
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing critical API keys in environment.")

# Configure low-level clients
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "drap-chatbot"
pinecone_index = pc.Index(INDEX_NAME)
co = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0 #
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=768
)

# ==========================================
# 2. Define Atomic (Granular) Tools
# ==========================================


@tool
def retrieve_documents(sub_queries: Annotated[List[str], "A list of distinct, simplified search query strings"]) -> List[str]:
    """
    Use this tool to search the Pinecone vector database for matching regulatory documents.
    If the user's question is complex, break it down into 2 to 3 simplified search phrases 
    and pass them ALL into this tool as a list.
    Returns a list of raw document chunks.
    """
    print(f"\n[System: Tool Activated] Retrieving documents for {len(sub_queries)} queries...")
    all_results = []
    seen_ids = set()
    
    for sq in sub_queries:
        vector = embeddings_model.embed_query(sq)
        query_res = pinecone_index.query(
            vector=vector,
            top_k=20,
            include_metadata=True
        )
        
        for match in query_res['matches']:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                all_results.append(match['metadata']['text'])
                
    return all_results


# ==========================================
# 3. Assemble the Autonomous LangGraph Agent
# ==========================================
# Provide the toolkit to the agent
tools_list = [retrieve_documents]

memory_saver = InMemorySaver()

# Build the Agent
agent_executor = create_agent(
    model=llm,
    tools=tools_list,
    checkpointer=memory_saver,
    system_prompt = """You are an expert AI assistant specialized in analyzing organizational meeting minutes. 
You have access to a vector database of DRAP regulatory records. 

For complex regulatory questions, you should autonomously plan your execution:
1. Before searching, analyze the question. If it is complex, multi-part, or vague, internally break it down into up to 3 distinct, simpler search phrases.
2. Pass your generated search phrases as a list into the `retrieve_documents` tool to pull raw data.
3. Finally, synthesize the answer using ONLY the retrieved context.

If the question is extremely simple (e.g., "What does DRAP stand for?"), you may pass a single query in the retrieval tool 
If a user greets you casually, do not invoke any tools—simply respond natively."""
)

# ==========================================
# 4. Interactive Live Loop
# ==========================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "drap_session_002"}}
    
    print("\n" + "="*60)
    print(" Fully Autonomous DRAP Agent Online")
    print("="*60)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down conversational engine.")
            break
            
        if not user_input.strip():
            continue
            
        try:
            response_state = agent_executor.invoke(
                {"messages": [("user", user_input)]}, 
                config=config
            )
            
            final_message = response_state["messages"][-1].content
            
            if isinstance(final_message, list):
                clean_reply = "".join(block["text"] for block in final_message if "text" in block)
            else:
                clean_reply = final_message
                
            print(f"\nAI: {clean_reply.strip()}")
            
        except Exception as e:
            print(f"\n[System Error]: {e}")
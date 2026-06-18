import os
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Native SDK imports retained for internal Pinecone/Cohere logic
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

# Configure low-level Pinecone and Cohere clients
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "drap-chatbot"
pinecone_index = pc.Index(INDEX_NAME)

co = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None

# Initialize LangChain's Gemini LLM wrapper (The Agent Brain)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0 # Keep temperature at 0 for factual accuracy
)

# Initialize LangChain's Gemini Embeddings wrapper
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    output_dimensionality = 768
)

# ==========================================
# 2. Define the Autonomous Tool
# ==========================================
@tool
def search_drap_minutes(query: str) -> str:
    """
    Use this tool to search through the Drug Regulatory Authority of Pakistan (DRAP) 
    Registration Board meeting minutes. Use it whenever the user asks questions about 
    specific drug approvals, manufacturer changes, pharmaceutical company applications, 
    age limits, combo packs, or regulatory decisions. Input must be a clear search query.
    """
    print(f"\n[System: Tool Activated] Scanning database for: '{query}'...")
    
    # --- Step A: Multi-Query Decomposition ---
    decomposition_prompt = f"""
    Break down the following complex question into up to 3 small, simplified queries to broaden search coverage.
    Output each query on a new line. Do not include numbers, bullet points, or extra text.
    Question: "{query}"
    """
    decomposed_res = llm.invoke(decomposition_prompt)
    sub_queries = [q.strip() for q in decomposed_res.content.strip().split('\n') if q.strip()]
    if not sub_queries:
        sub_queries = [query]
        
    # --- Step B: Vector Retrieval from Pinecone ---
    all_results = []
    seen_ids = set()
    
    for sq in sub_queries:
        # Generate 768-dimensional vector
        vector = embeddings_model.embed_query(sq)
        
        query_res = pinecone_index.query(
            vector=vector,
            top_k=20,
            include_metadata=True
        )
        
        for match in query_res['matches']:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                all_results.append({
                    "text": match['metadata']['text'],
                    "score": match['score']
                })
                
    if not all_results:
        return "No matching regulatory records found in the database."

    # --- Step C: Context Reranking via Cohere ---
    docs_text = [res['text'] for res in all_results]
    
    if co:
        try:
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_text,
                top_n=10 
            )
            final_contexts = [docs_text[r.index] for r in rerank_response.results]
        except Exception as e:
            print(f"  -> Cohere Rerank failed: {e}. Falling back to raw results.")
            final_contexts = docs_text[:10]
    else:
        final_contexts = docs_text[:10]
        
    # Combine the top 10 chunks into a single string to hand back to the Agent
    return "\n\n--- REGULATORY RECORD FRAGMENT ---\n\n".join(final_contexts)


# ==========================================
# 3. Assemble the LangGraph Agent
# ==========================================
# Pack our single database tool into a list
tools_list = [search_drap_minutes]

# Initialize persistent chat memory
memory_saver = InMemorySaver()

# Build the Agent 
agent_executor = create_agent(
    model=llm,
    tools=tools_list,
    checkpointer=memory_saver,
    system_prompt = """You are an expert AI assistant specialized in analyzing organizational meeting minutes. 
When answering questions using your search tool, maintain absolute factual accuracy based *only* on the tool outputs. 
If a user greets you casually, do not invoke your tool—simply respond natively in a friendly manner."""
)

# ==========================================
# 4. Interactive Live Loop
# ==========================================
if __name__ == "__main__":
    # Assign a thread_id so the memory engine tracks this specific conversation
    config = {"configurable": {"thread_id": "drap_session_001"}}
    
    print("\n" + "="*60)
    print(" Agentic DRAP RAG System Online (LangGraph Framework)")
    print("="*60)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down conversational engine.")
            break
            
        if not user_input.strip():
            continue
            
        try:
            # Invoke the LangGraph agent
            response_state = agent_executor.invoke(
                {"messages": [("user", user_input)]}, 
                config=config
            )
            
            # Extract and print the final message from the AI
            final_message = response_state["messages"][-1].content
            
            # Clean it up: If Gemini returns a list of blocks, extract only the text
            if isinstance(final_message, list):
                clean_reply = "".join(block["text"] for block in final_message if "text" in block)
            else:
                clean_reply = final_message
                
            print(f"\nAI: {clean_reply.strip()}")
            
        except Exception as e:
            print(f"\n[System Error]: {e}")
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
import cohere

# Load environment variables
load_dotenv()

# Initialize API Keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("Warning: GEMINI_API_KEY or PINECONE_API_KEY is not set. Execution will fail when pipeline runs.")

# Configure Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "drap-chatbot"
    try:
        pinecone_index = pc.Index(INDEX_NAME)
    except Exception as e:
        pinecone_index = None
else:
    pinecone_index = None

if COHERE_API_KEY:
    co = cohere.Client(api_key=COHERE_API_KEY)
else:
    co = None

# Default models
LLM_MODEL = "gemini-2.5-flash"  # Using flash for fast generation tasks
EMBEDDING_MODEL = "models/gemini-embedding-001"

def agent_router(query):
    """
    Evaluates what is being asked and routes the execution path.
    """
    print("\n[Router] Evaluating Query...")
    prompt = f"""
    You are a classification router. Evaluate the following user query.
    If the query is a basic conversational greeting (like "hello", "hi", "hey"), output EXACTLY and ONLY the word "SIMPLE".
    If the query suggests requesting domain information, facts, or answering a question, output EXACTLY and ONLY the word "COMPLEX".
    
    Query: "{query}"
    """
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    decision = response.text.strip().upper()
    
    if "SIMPLE" in decision:
        return "SIMPLE"
    return "COMPLEX"

def handle_simple_branch(query):
    """
    Branch 1: Simple/Clarity
    """
    print("[Branch 1] Routing to simple conversational block...")
    prompt = f"The user said '{query}'. Respond with a simple conversational variation of 'hey'."
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

def multi_query_decomposition(query):
    """
    Branch 2A: Multi-Query Decomposition
    """
    print("[Branch 2A] Decomposing complex query...")
    prompt = f"""
    Break down the following complex question into up to 3 small, simplified queries to broaden search coverage.
    Output each query on a new line. Do not include numbers, bullet points, or extra text.
    
    Question: "{query}"
    """
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
    if not queries:
        queries = [query]
    print(f"  -> Decomposed queries: {queries}")
    return queries

def retrieve_context(queries):
    """
    Branch 2B: Retriever
    """
    print("[Branch 2B] Vectorizing queries and executing similarity search...")
    all_results = []
    seen_ids = set()
    
    for q in queries:
        # Vectorize using gemini-embedding-001 and truncate to 768
        embedding_response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=q,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        vector = embedding_response['embedding']
        
        # Search via Pinecone, Top k=20
        # Dotproduct metric was configured at index creation in Phase 1
        query_res = pinecone_index.query(
            vector=vector,
            top_k=20,
            include_metadata=True
        )
        
        for match in query_res['matches']:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                # Store the raw text and its original Pinecone score
                all_results.append({
                    "id": match['id'],
                    "text": match['metadata']['text'],
                    "score": match['score']
                })
    
    print(f"  -> Retrieved {len(all_results)} unique context fragments across all sub-queries.")
    return all_results

def local_sorting_placeholder(query, documents):
    """
    Local Python sorting function if Cohere API key is missing.
    Simply sorts by length of the context string as a basic placeholder logic.
    """
    print("  -> (Using local Python sorting placeholder instead of Cohere)")
    # Basic dummy sort: count occurrences of words from query in text
    query_words = set(query.lower().split())
    
    def score_doc(doc):
        doc_words = doc['text'].lower().split()
        return sum(1 for word in query_words if word in doc_words)

    # Sort descending by custom score, then by original pinecone score
    sorted_docs = sorted(documents, key=lambda d: (score_doc(d), d['score']), reverse=True)
    return sorted_docs

def context_builder_rerank(original_query, retrieved_results):
    """
    Branch 2C: Context Builder
    """

    print("[Branch 2C] Bypassing Reranker completely - passing raw Pinecone results.")
    return retrieved_results #skipping reranking for testing 
    print("[Branch 2C] Aggregating results and passing through Reranker...")
    if not retrieved_results:
        return []

    if COHERE_API_KEY:
        print("  -> Using Cohere Rerank API...")
        docs_text = [res['text'] for res in retrieved_results]
        try:
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=original_query,
                documents=docs_text,
                top_n=len(docs_text)
            )
            
            reranked_results = []
            for result in rerank_response.results:
                reranked_results.append({
                    "text": docs_text[result.index],
                    "score": result.relevance_score
                })
            return reranked_results
        except Exception as e:
            print(f"  -> Cohere Rerank failed: {e}. Falling back to local sort.")
            return local_sorting_placeholder(original_query, retrieved_results)
    else:
        return local_sorting_placeholder(original_query, retrieved_results)

def final_generation(original_query, reranked_contexts):
    """
    Final Step: Generation
    """
    print("[Final Step] Converting context fragments into a cohesive response...")
    
    context_text = "\n\n---CONTEXT FRAGMENT---\n\n".join(
        [item['text'] for item in reranked_contexts[:10]] # Pass the top 10 reranked to avoid token overload natively
    )
    
    prompt = f"""
    You are an AI assistant answering questions about organizational meeting minutes.
    Use the provided context fragments to answer the user's question accurately.
    If the context does not contain the answer, state that you do not have enough information.
    
    User Question: "{original_query}"
    
    Context:
    {context_text}
    """
    
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

def run_query_pipeline(user_query):
    print(f"\n{'='*50}\nIncoming Query: '{user_query}'\n{'='*50}")
    
    # Step 2. Agent Router
    route = agent_router(user_query)
    
    if route == "SIMPLE":
        # Branch 1
        response = handle_simple_branch(user_query)
        print("\n[AI Response] (Simple Route):")
        print(response)
    
    elif route == "COMPLEX":
        if pinecone_index is None:
             print("Error: Pinecone index not initialized. Check PINECONE_API_KEY.")
             return
             
        # Branch 2A
        decomposed_queries = multi_query_decomposition(user_query)
        
        # Branch 2B
        retrieved_results = retrieve_context(decomposed_queries)
        
        # Branch 2C
        reranked_results = context_builder_rerank(user_query, retrieved_results)
        
        # Step 3
        response = final_generation(user_query, reranked_results)
        print("\n[AI Response] (Complex Route):")
        print(response)

if __name__ == "__main__":
    # Test cases showing the bottlenecks exactly as designed in the architecture.
    while True:
        user_input = input("\nEnter a query (or type 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit']:
            break
        try:
            run_query_pipeline(user_input)
        except Exception as e:
            print(f"Pipeline Execution Error: {e}")

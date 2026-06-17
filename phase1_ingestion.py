import os
import time
from pypdf import PdfReader
import tiktoken
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables (Make sure to create a .env file with PINECONE_API_KEY and GEMINI_API_KEY)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys. Please set PINECONE_API_KEY and GEMINI_API_KEY in your environment or .env file.")

genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "drap-chatbot"

def setup_pinecone():
    print("Checking Pinecone index...")
    # Check if index exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Serverless Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, # Must match Gemini's text-embedding-004 output dimension
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            print("Waiting for index to be ready...")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    return pc.Index(INDEX_NAME)

def load_documents(directory="dra_meeting_minutes"):
    print(f"Loading documents from {directory}...")
    documents = []
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return documents

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
                documents.append({"id": filename, "text": text})
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    
    print(f"Loaded {len(documents)} PDF files.")
    return documents

def chunk_text(text, max_tokens=512, overlap=50):
    """
    Token-based chunking mechanism using tiktoken.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        # Slide window with overlap
        start += (max_tokens - overlap)
        
    return chunks

def embed_and_upsert(index, documents):
    print("Chunking, embedding, and upserting data...")
    # The correct model name format for the deprecated package is usually text-embedding-004
    # The SDK automatically handles the models/ prefix logic in newer versions, but we need
    # to format it correctly for this SDK version. 
    model = "models/gemini-embedding-001"
    
    batch_size = 100 # Batch size for Pinecone upsert
    vectors = []
    
    for doc in documents:
        print(f"Processing document: {doc['id']}")
        chunks = chunk_text(doc["text"], max_tokens=512, overlap=50)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": doc["id"],
                "chunk_id": i,
                "text": chunk  # Upserting the raw text chunk string as metadata
            }
            
            try:
                # Generate embedding using Gemini
                response = genai.embed_content(
                    model=model,
                    content=chunk,
                    task_type="retrieval_document",
                    output_dimensionality=768
                )
                embedding = response['embedding']
                
                # Format: (id, values, metadata)
                vector_id = f"{doc['id']}_chunk_{i}"
                vectors.append((vector_id, embedding, chunk_metadata))
                
                # Upsert into Pinecone when batch size is reached
                if len(vectors) >= batch_size:
                    index.upsert(vectors=vectors)
                    print(f"Upserted {len(vectors)} vectors...")
                    vectors = []
                    
            except Exception as e:
                print(f"Embedding/Upsert failed for chunk {i} of {doc['id']}. Error: {e}")
                time.sleep(2) # Backoff in case of API limits

    # Upsert the remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Upserted final {len(vectors)} vectors.")
        
    print("Ingestion complete.")

def main():
    index = setup_pinecone()
    documents = load_documents()
    if documents:
        embed_and_upsert(index, documents)

if __name__ == "__main__":
    main()

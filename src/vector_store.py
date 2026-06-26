import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Verify that the API key is configured
if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "YOUR_GEMINI_API_KEY_HERE":
    print("[WARNING] GEMINI_API_KEY is not set or is still a placeholder in the .env file.")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

def get_embeddings():
    """
    Initialize and return a free local HuggingFace embedding model.
    This runs entirely locally on CPU/GPU and bypasses API key restrictions.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    """
    Load markdown documents, split them into chunks, and index them in Chroma DB.
    """
    import shutil
    from langchain_core.documents import Document
    
    print("--- Starting Document Ingestion ---")
    
    # Clear existing vector store database directory to prevent duplicates on rebuild
    if CHROMA_DIR.exists():
        print(f"Clearing existing vector store at {CHROMA_DIR}...")
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception as e:
            print(f"[WARNING] Could not remove directory {CHROMA_DIR}: {e}")
            
    # 1. Load documents
    documents = []
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory {DATA_DIR} does not exist.")
        return
        
    md_files = list(DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"[WARNING] No markdown files found in {DATA_DIR}.")
        return

    for file_path in md_files:
        print(f"Loading document: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Store content and metadata (filename)
            documents.append({
                "page_content": text,
                "metadata": {"source": file_path.name}
            })

    # 2. Split documents into chunks
    # We use chunk_size=1000 characters and chunk_overlap=200 characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["page_content"])
        for chunk_text in split_texts:
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=doc["metadata"]
                )
            )

    print(f"Total chunks created: {len(chunks)}")

    # 3. Vectorize and save to Chroma
    print(f"Saving vector store index to local folder: {CHROMA_DIR}...")
    embeddings = get_embeddings()
    
    # Initialize Chroma database with the chunk documents
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    print("--- Document Ingestion Completed Successfully! ---")
    return db

def get_retriever(k=3):
    """
    Return a retriever object to query the existing Chroma DB.
    """
    embeddings = get_embeddings()
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    return db.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    # Test script: allows manual execution to build/rebuild vector DB
    try:
        db = build_vector_store()
        
        # Test query
        if db:
            query = "What is considered a high-risk AI system under the EU AI Act?"
            print(f"\nTesting retriever with query: '{query}'")
            retriever = get_retriever(k=2)
            results = retriever.invoke(query)
            print(f"Retrieved {len(results)} relevant documents:")
            for i, doc in enumerate(results):
                print(f"\n[Document {i+1} (Source: {doc.metadata.get('source')})]:")
                print(doc.page_content[:200] + "...")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during database build: {e}")
        print("Please check your GEMINI_API_KEY in the .env file.")

import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def extract_data(file_path):
    """
    Extracts data from a CSV file and converts each row to a Document object.
    """
    try:
        df = pd.read_csv(file_path)
        documents = []
        for _, row in df.iterrows():
            # Construct the text content
            text = f"Title: {row['title']}\nGenres: {row['Genres']}\nSynopsis: {row['description']}\nThemes: {row['Themes']}"
            
            # Create a Document object
            # We can store metadata as well if needed, e.g., the ID or Score
            metadata = {
                "id": str(row['myanimelist_id']),
                "title": row['title'],
                "score": row['Score'],
                "genres": row['Genres'],
                "episodes": row['Episodes'],
                "image_url": row['image'],
                "rating": row['Rating'],
                "demographic": row['Demographic']
            }
            # Clean up metadata to remove NaN values which Pinecone might not like
            metadata = {k: v for k, v in metadata.items() if pd.notna(v)}
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        print(f"Extracted {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        print(f"Error extracting data: {e}")
        return []

def get_embeddings():
    """
    Initializes and returns Hugging Face embeddings.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return None

def ingest_embeddings(documents, index_name):
    """
    Ingests documents into a Pinecone index using the provided embeddings.
    """
    try:
        embeddings = get_embeddings()
        if not embeddings:
            return

        # Initialize Pinecone
        # Ensure PINECONE_API_KEY is set in environment variables
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        # Check if index exists, if not create it (optional, or assume it exists)
        # For this task, we'll assume we might need to create it or just connect
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name in existing_indexes:
            print(f"Index '{index_name}' exists. Checking if it needs update...")
            # For now, we will just print a message. In a real scenario, we might check doc counts or force update.
            # But the user asked to ingest, so we might want to overwrite or add.
            # Since we are changing schema, it is safer to delete and recreate or just add.
            # Let's just add for now, but usually one would want to clear old data if schema changes drastically.
            # However, the user didn't explicitly ask to delete. I'll proceed with adding.
            pass

        if index_name not in existing_indexes:
            print(f"Creating index {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=384, # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
        
        print(f"Ingesting {len(documents)} documents into index '{index_name}'...")
        
        # Using LangChain's PineconeVectorStore to ingest
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )
        print("Ingestion complete.")

    except Exception as e:
        print(f"Error ingesting embeddings: {e}")

if __name__ == "__main__":
    # Example usage
    DATA_PATH = "Data/mal_anime.csv"
    INDEX_NAME = "anime-recommendation-v2"
    
    if os.path.exists(DATA_PATH):
        docs = extract_data(DATA_PATH)
        if docs:
            # Ingest only a subset for testing if needed, or all. 
            # The file is large (19k lines), might take a while. 
            # I'll ingest all as requested.
            ingest_embeddings(docs, INDEX_NAME)
            
    else:
        print(f"File not found: {DATA_PATH}")

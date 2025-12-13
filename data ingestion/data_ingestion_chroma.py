import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def extract_data(file_path):
    """
    Extracts data from a CSV file and converts each row to a Document object.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

        df = pd.read_csv(file_path)
        documents = []
        for _, row in df.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else "Unknown"
            genres = str(row['Genres']) if pd.notna(row['Genres']) else "Unknown"
            description = str(row['description']) if pd.notna(row['description']) else "No description available"
            themes = str(row['Themes']) if pd.notna(row['Themes']) else "Unknown"

            text = f"Title: {title}\nGenres: {genres}\nSynopsis: {description}\nThemes: {themes}"
            
            metadata = {
                "id": str(row['myanimelist_id']),
                "title": title,
                "score": float(row['Score']) if pd.notna(row['Score']) else 0.0,
                "genres": genres,
                "episodes": int(row['Episodes']) if pd.notna(row['Episodes']) and str(row['Episodes']).isdigit() else 0,
                "image_url": str(row['image']) if pd.notna(row['image']) else "",
                "rating": str(row['Rating']) if pd.notna(row['Rating']) else "Unknown",
                "demographic": str(row['Demographic']) if pd.notna(row['Demographic']) else "Unknown"
            }
            # Clean up metadata? Chroma handles nulls okay usually, but simpler types are safer.
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        print(f"Extracted {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        print(f"Error extracting data: {e}")
        return []

def get_embeddings():
    """
    Initializes and returns HuggingFace embeddings.
    """
    try:
        print("Initializing HuggingFace Embeddings...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return None

def ingest_to_chroma(documents):
    """
    Ingests documents into a Chroma vectorstore using HuggingFace embeddings.
    """
    try:
        embeddings = get_embeddings()
        if not embeddings:
            return

        persist_directory = "./chroma_db"
            
        print(f"Ingesting {len(documents)} documents into Chroma DB at '{persist_directory}' using HuggingFace embeddings...")
        
        # Create or update the vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        print("Ingestion complete.")
        return vectorstore

    except Exception as e:
        print(f"Error ingesting into Chroma: {e}")

if __name__ == "__main__":
    DATA_PATH = "Data/mal_anime.csv"
    
    docs = extract_data(DATA_PATH)
    if docs:
        ingest_to_chroma(docs)

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

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

def retrieve_anime_recommendations(query: str, k: int = 5):
    """
    Performs semantic search in Pinecone to get top k anime recommendations.
    
    Args:
        query (str): The user's search query.
        k (int): Number of recommendations to return.
        
    Returns:
        list: A list of matched documents.
    """
    try:
        embeddings = get_embeddings()
        if not embeddings:
            return []

        index_name = "anime-recommendation"
        
        # Initialize PineconeVectorStore
        # We assume PINECONE_API_KEY is in the environment variables
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        
        results = vectorstore.similarity_search(query, k=k)
        return results
        
    except Exception as e:
        print(f"Error in retrieve_anime_recommendations: {e}")
        return []

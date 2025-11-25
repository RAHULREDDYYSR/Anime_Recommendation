import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Try to import streamlit for caching, fallback to manual caching if not available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Global cache for non-Streamlit environments
_embeddings_cache = None
_vectorstore_cache = None

def get_embeddings():
    """
    Returns cached Hugging Face embeddings (singleton).
    Uses Streamlit's @st.cache_resource if available, otherwise uses global cache.
    """
    if HAS_STREAMLIT:
        return _get_embeddings_streamlit()
    else:
        return _get_embeddings_global()

@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embeddings_streamlit():
    """Streamlit-cached version"""
    print("Initializing embedding model (cached by Streamlit)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded successfully!")
    return embeddings

def _get_embeddings_global():
    """Global cache version for non-Streamlit environments"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        print("Initializing embedding model (first time only)...")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded successfully!")
    
    return _embeddings_cache

def get_vectorstore():
    """
    Returns cached Pinecone vectorstore (singleton).
    Uses Streamlit's @st.cache_resource if available, otherwise uses global cache.
    """
    if HAS_STREAMLIT:
        return _get_vectorstore_streamlit()
    else:
        return _get_vectorstore_global()

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def _get_vectorstore_streamlit():
    """Streamlit-cached version"""
    embeddings = get_embeddings()
    if not embeddings:
        return None
    
    index_name = "anime-recommendation-v2"
    print(f"Connecting to Pinecone index '{index_name}' (cached by Streamlit)...")
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    print("Pinecone connection established!")
    return vectorstore

def _get_vectorstore_global():
    """Global cache version for non-Streamlit environments"""
    global _vectorstore_cache
    
    if _vectorstore_cache is None:
        embeddings = get_embeddings()
        if not embeddings:
            return None
        
        index_name = "anime-recommendation-v2"
        print(f"Connecting to Pinecone index '{index_name}' (first time only)...")
        
        _vectorstore_cache = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        print("Pinecone connection established!")
    
    return _vectorstore_cache

def retrieve_anime_recommendations(query: str, k: int = 5):
    """
    Performs semantic search in Pinecone to get top k anime recommendations.
    Uses cached embeddings and vectorstore for fast retrieval.
    
    Args:
        query (str): The user's search query.
        k (int): Number of recommendations to return.
        
    Returns:
        list: A list of matched documents.
    """
    try:
        vectorstore = get_vectorstore()
        if not vectorstore:
            return []
        
        results = vectorstore.similarity_search(query, k=k)
        return results
        
    except Exception as e:
        print(f"Error in retrieve_anime_recommendations: {e}")
        return []

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
_vectorstore_cache_pinecone = None
_vectorstore_cache_chroma = None

def get_embeddings():
    """
    Returns cached HuggingFace embeddings (singleton).
    """
    if HAS_STREAMLIT:
        return _get_embeddings_streamlit()
    else:
        return _get_embeddings_global()

@st.cache_resource(show_spinner="Loading HuggingFace embedding model...")
def _get_embeddings_streamlit():
    """Streamlit-cached version (HuggingFace)"""
    print("Initializing HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("HuggingFace embedding model loaded successfully!")
    return embeddings

def _get_embeddings_global():
    """Global cache version for non-Streamlit environments (HuggingFace)"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        print("Initializing HuggingFace embedding model (first time only)...")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("HuggingFace embedding model loaded successfully!")
    
    return _embeddings_cache

def get_vectorstore(source: str = "Pinecone"):
    """
    Returns cached vectorstore (singleton) based on source.
    Always uses HuggingFace embeddings.
    """
    if HAS_STREAMLIT:
        if source == "ChromaDB":
             return _get_vectorstore_chroma_streamlit()
        return _get_vectorstore_pinecone_streamlit()
    else:
        if source == "ChromaDB":
             return _get_vectorstore_chroma_global()
        return _get_vectorstore_pinecone_global()

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def _get_vectorstore_pinecone_streamlit():
    """Streamlit-cached version for Pinecone with HuggingFace embeddings"""
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

@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def _get_vectorstore_chroma_streamlit():
    """Streamlit-cached version for ChromaDB with HuggingFace embeddings"""
    embeddings = get_embeddings()
    if not embeddings:
        return None
    
    persist_directory = "./chroma_db"
    print(f"Connecting to ChromaDB at '{persist_directory}' (cached by Streamlit)...")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("ChromaDB connection established!")
    return vectorstore

def _get_vectorstore_pinecone_global():
    """Global cache version for non-Streamlit environments (Pinecone)"""
    global _vectorstore_cache_pinecone
    
    if _vectorstore_cache_pinecone is None:
        embeddings = get_embeddings()
        if not embeddings:
            return None
            
        index_name = "anime-recommendation-v2"
        print(f"Connecting to Pinecone index '{index_name}' (first time only)...")
        
        _vectorstore_cache_pinecone = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        print("Pinecone connection established!")
    
    return _vectorstore_cache_pinecone

def _get_vectorstore_chroma_global():
    """Global cache version for non-Streamlit environments (ChromaDB)"""
    global _vectorstore_cache_chroma
    
    if _vectorstore_cache_chroma is None:
        embeddings = get_embeddings()
        if not embeddings:
            return None
            
        persist_directory = "./chroma_db"
        print(f"Connecting to ChromaDB at '{persist_directory}' (first time only)...")
        
        _vectorstore_cache_chroma = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("ChromaDB connection established!")
    
    return _vectorstore_cache_chroma

def retrieve_anime_recommendations(query: str, k: int = 5, source: str = "Pinecone"):
    """
    Performs semantic search to retrieve top k anime recommendations.
    Uses cached HuggingFace embeddings and vectorstore based on source.
    """
    try:
        vectorstore = get_vectorstore(source=source)
        if not vectorstore:
            return []
        
        results = vectorstore.similarity_search(query, k=k)
        return results
        
    except Exception as e:
        print(f"Error in retrieve_anime_recommendations: {e}")
        return []

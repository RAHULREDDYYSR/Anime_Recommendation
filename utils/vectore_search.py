import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
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
# Global cache for non-Streamlit environments
_embeddings_cache_hf = None
_embeddings_cache_openai = None
_vectorstore_cache_pinecone = None
_vectorstore_cache_chroma = None

def get_embeddings(model_source: str = "HuggingFace"):
    """
    Returns cached embeddings (singleton) based on source.
    """
    if HAS_STREAMLIT:
        if model_source == "OpenAI":
            return _get_embeddings_openai_streamlit()
        return _get_embeddings_streamlit()
    else:
        if model_source == "OpenAI":
            return _get_embeddings_openai_global()
        return _get_embeddings_global()

@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embeddings_streamlit():
    """Streamlit-cached version (HuggingFace)"""
    print("Initializing embedding model (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded successfully!")
    return embeddings

@st.cache_resource(show_spinner="Loading OpenAI embeddings...")
def _get_embeddings_openai_streamlit():
    """Streamlit-cached version (OpenAI)"""
    print("Initializing embedding model (OpenAI)...")
    # Assuming OPENAI_API_KEY is in .env
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding model loaded successfully!")
    return embeddings

def _get_embeddings_global():
    """Global cache version for non-Streamlit environments (HF)"""
    global _embeddings_cache_hf
    
    if _embeddings_cache_hf is None:
        print("Initializing embedding model (first time only)...")
        _embeddings_cache_hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded successfully!")
    
    return _embeddings_cache_hf

def _get_embeddings_openai_global():
    """Global cache version for non-Streamlit environments (OpenAI)"""
    global _embeddings_cache_openai
    
    if _embeddings_cache_openai is None:
        print("Initializing embedding model (OpenAI - first time only)...")
        _embeddings_cache_openai = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Embedding model loaded successfully!")
    
    return _embeddings_cache_openai

def get_vectorstore(source: str = "Pinecone", embedding_model: str = "HuggingFace"):
    """
    Returns cached vectorstore (singleton) based on source and embedding model.
    """
    if HAS_STREAMLIT:
        if source == "ChromaDB":
             return _get_vectorstore_chroma_streamlit(embedding_model)
        return _get_vectorstore_pinecone_streamlit(embedding_model)
    else:
        # Global caching for all combinations logic is becoming complex.
        # Simplification: pass embedding_model to specific getters
        if source == "ChromaDB":
             return _get_vectorstore_chroma_global(embedding_model)
        return _get_vectorstore_pinecone_global(embedding_model)

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def _get_vectorstore_pinecone_streamlit(embedding_model: str = "HuggingFace"):
    """Streamlit-cached version"""
    embeddings = get_embeddings(embedding_model)
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
def _get_vectorstore_chroma_streamlit(embedding_model: str = "HuggingFace"):
    """Streamlit-cached version for ChromaDB"""
    embeddings = get_embeddings(embedding_model)
    if not embeddings:
        return None
    
    # We might want separate directories for separate models if they share the same 'chroma_db' root but different collections.
    # However, Chroma usually manages collections. Here we are using persist_directory which is the DB root.
    # If the user switches models for the *same* DB, retrieving logic might fail if dimensions differ.
    if embedding_model == "OpenAI":
        persist_directory = "./chroma_db_openai"
    else:
        persist_directory = "./chroma_db"
        
    print(f"Connecting to ChromaDB at '{persist_directory}' (cached by Streamlit)...")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("ChromaDB connection established!")
    return vectorstore

def _get_vectorstore_pinecone_global(embedding_model: str = "HuggingFace"):
    """Global cache version for non-Streamlit environments (Pinecone)"""
    # Note: Simplification - we are overwriting the global cache if model changes for simplicity in this script
    global _vectorstore_cache_pinecone
    
    # In a real app we might want a dict cache keyed by model.
    # For now, let's assume we just rebuild if needed or use what's there (risk of mismatch if using global cache for multiple models sequentially).
    # Given the use case is single user local run, we can just fetch fresh.
    
    embeddings = get_embeddings(embedding_model)
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

def _get_vectorstore_chroma_global(embedding_model: str = "HuggingFace"):
    """Global cache version for non-Streamlit environments (ChromaDB)"""
    # Simply returning a new instance or cached one.
    global _vectorstore_cache_chroma
    
    # See note above about global cache invalidation.
    embeddings = get_embeddings(embedding_model)
    if not embeddings:
        return None
        
    if embedding_model == "OpenAI":
        persist_directory = "./chroma_db_openai"
    else:
        persist_directory = "./chroma_db"
        
    print(f"Connecting to ChromaDB at '{persist_directory}' (first time only)...")
    
    _vectorstore_cache_chroma = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("ChromaDB connection established!")
    
    return _vectorstore_cache_chroma

def retrieve_anime_recommendations(query: str, k: int = 5, source: str = "Pinecone", embedding_model: str = "HuggingFace"):
    """
    Performs semantic search to retrieve top k anime recommendations.
    Uses cached embeddings and vectorstore based on source and model.
    """
    try:
        vectorstore = get_vectorstore(source=source, embedding_model=embedding_model)
        if not vectorstore:
            return []
        
        results = vectorstore.similarity_search(query, k=k)
        return results
        
    except Exception as e:
        print(f"Error in retrieve_anime_recommendations: {e}")
        return []

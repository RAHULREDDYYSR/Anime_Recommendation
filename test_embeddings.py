from langchain_huggingface import HuggingFaceEmbeddings
import time

print("Starting embedding test...")
try:
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"Successfully loaded embeddings in {time.time() - start:.2f}s")
    
    # Test a simple embedding
    vec = embeddings.embed_query("test")
    print(f"Embedding generated, length: {len(vec)}")
    
except Exception as e:
    print(f"FAILED to load embeddings: {e}")

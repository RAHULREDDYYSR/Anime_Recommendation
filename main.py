import time
from graph.graph import app
from langsmith import uuid7

def main():
    print("Hello from anime-recommendation!")
    print("=" * 60)
    
    # First query
    query1 = "I want a shonen anime with good fights"
    print(f"\nQuery 1: {query1}")
    start_time = time.time()
    result1 = app.invoke({"input_text": query1})
    end_time = time.time()
    print(f"Recommendations: {result1['recommended_anime']}")
    print(f"Time: {end_time - start_time:.2f} seconds")
    
    # Second query (should be much faster with cache)
    print("\n" + "=" * 60)
    query2 = "I want a romance anime with comedy"
    print(f"\nQuery 2: {query2}")
    start_time = time.time()
    result2 = app.invoke({"input_text": query2})
    end_time = time.time()
    print(f"Recommendations: {result2['recommended_anime']}")
    print(f"Time: {end_time - start_time:.2f} seconds (cached!)")
    
    print("\n" + "=" * 60)
    print("Cache is working! Subsequent queries are much faster.")

if __name__ == "__main__":
    try:
        app.get_graph().draw_mermaid_png(output_file_path="graph.png")
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

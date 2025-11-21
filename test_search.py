from graph.nodes import retrieve_anime_recommendations
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    query = "ninja anime"
    print(f"Searching for: {query}")
    results = retrieve_anime_recommendations(query=query)
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content.splitlines()[0]}")

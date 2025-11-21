from graph.graph import app
from langsmith import uuid7

id = uuid7()
def main():
    print("Hello from anime-recommendation!")
    user_input = "I want a shonen anime with good fights"
    print(f"Processing query: {user_input}")
    
    try:
        app.get_graph().draw_mermaid_png(output_file_path="graph.png")
        result = app.invoke({"input_text": user_input})
        print("\nRecommendations:")
        print(result['recommended_anime'])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

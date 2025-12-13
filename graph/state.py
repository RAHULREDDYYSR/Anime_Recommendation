from typing import Optional, TypedDict, List

class GraphState(TypedDict):
    '''Defines the state for our LangGraph workflow'''
    input_text: str
    redefine_input_content: str
    recommended_anime: List[dict] # Changed from List[str] to List[dict] to hold AnimeDetails
    context: List[str]
    vector_source: str # "Pinecone" or "ChromaDB"
    llm_model: str # "Groq", "Gemini", or "OpenAI"
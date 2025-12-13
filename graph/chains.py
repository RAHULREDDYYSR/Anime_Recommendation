from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import RecommendedAnime, RefinedQuery
from langchain_openai import ChatOpenAI


def get_llm_chains(model_name: str = "Groq"):
    """
    Returns the LLM chains for refine_input and recommended_anime based on the selected model.
    """
    if model_name == "Gemini":
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
    elif model_name == "OpenAI":
        llm = ChatOpenAI(model='gpt-4o-mini') # Using a standard model name, user can adjust
    else: # Default to Groq
        llm = ChatGroq(model='llama-3.3-70b-versatile')

    redefine_input_llm = llm.with_structured_output(RefinedQuery)
    recommended_Anime_llm = llm.with_structured_output(RecommendedAnime)
    
    return redefine_input_llm, recommended_Anime_llm
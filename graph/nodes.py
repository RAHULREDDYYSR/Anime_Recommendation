from .state import GraphState
from .chains import recommended_Anime_llm, redefine_input_llm
from utils.vectore_search import retrieve_anime_recommendations

def redefine_input(state: GraphState) -> GraphState:
    prompt = f""" analyze the user's raw input and rewrite it into a precise, detailed description of what they are looking for.\n\n
    imput:\n ---\n {state['input_text']}
    """
    response = redefine_input_llm.invoke(prompt)
    state['redefine_input_content'] = response.refined_query
    return state

def anime_semantic_search(state: GraphState) -> GraphState:
    query = state['redefine_input_content']
    context = retrieve_anime_recommendations(query=query, k=5)
    state['context'] = context
    return state
    
def anime_recommendation(state: GraphState) -> GraphState:
    query = state['redefine_input_content']
    context = state['context']
    prompt = f"""You are an expert Anime Recommender. Recommend the best 3 anime titles based strictly on the Retrieved Context provided below that match the User’s Request..
    \n\n user's Request: {state['redefine_input_content']}.... \n\n --------\n retrived Context: {context}
    """
    state['recommended_anime'] = recommended_Anime_llm.invoke(prompt).anime_titles
    return state
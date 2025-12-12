from .state import GraphState
from .chains import recommended_Anime_llm, redefine_input_llm
from utils.vectore_search import retrieve_anime_recommendations
from langchain_core.messages import SystemMessage, HumanMessage

def redefine_input(state: GraphState) -> GraphState:
    """
    Analyzes and refines the user's raw input into a precise, detailed query.
    """
    messages = [
        SystemMessage(
            content="""You are an expert query refinement assistant specializing in anime recommendations.
Your task is to analyze the user's raw input and transform it into a precise, detailed, and well-structured description
that captures their preferences, interests, and requirements.

Focus on:
- Identifying key themes, genres, or characteristics mentioned
- Expanding abbreviations or shorthand references
- Clarifying vague or ambiguous terms
- Maintaining the user's original intent while improving clarity
- Creating a search-optimized query that will retrieve the most relevant anime recommendations"""
        ),
        HumanMessage(
            content=f"""Please refine the following user input into a detailed and precise query:

User Input: {state['input_text']}"""
        )
    ]
    
    response = redefine_input_llm.invoke(messages)
    state['redefine_input_content'] = response.refined_query
    return state

def anime_semantic_search(state: GraphState) -> GraphState:
    """
    Performs semantic search to retrieve relevant anime recommendations from the vector database.
    """
    query = state['redefine_input_content']
    context = retrieve_anime_recommendations(query=query, k=15)
    state['context'] = context
    return state
    
def anime_recommendation(state: GraphState) -> GraphState:
    """
    Generates final anime recommendations based on the refined query and retrieved context.
    """
    query = state['redefine_input_content']
    context = state['context']
    
    messages = [
        SystemMessage(
            content="""You are an expert anime recommendation specialist with deep knowledge of anime across all genres, demographics, and eras.

Your task is to analyze the retrieved anime data and recommend the top 10 anime titles that best match the user's preferences.

Guidelines:
- Recommend ONLY anime that appear in the Retrieved Context provided
- Select the 10 most relevant titles based on the user's refined query
- Prioritize quality matches over quantity - ensure recommendations truly align with user preferences
- Consider factors like genres, themes, scores, demographics, and descriptions when ranking
- Extract ALL required details accurately from the context for each recommendation

Required details for each anime:
1. Title - The official anime title
2. Description - Complete synopsis/description from the context
3. Score - The anime's rating score
4. Image URL - The image/poster URL
5. Episodes - Number of episodes
6. Rating - Age rating classification
7. Genres - All applicable genres
8. Demographic - Target demographic (e.g., Shounen, Seinen, Shoujo, etc.)"""
        ),
        HumanMessage(
            content=f"""Based on the user's preferences and the retrieved anime data, provide your top 10 recommendations.

User's Refined Query:
{query}

Retrieved Context:
{context}

Please extract and return the 10 best matching anime with all required details."""
        )
    ]
    
    # The LLM is already bound with the RecommendedAnime schema which contains the list of AnimeDetails
    state['recommended_anime'] = recommended_Anime_llm.invoke(messages).anime_titles
    return state
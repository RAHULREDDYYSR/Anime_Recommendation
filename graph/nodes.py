from .state import GraphState
from .chains import get_llm_chains
from utils.vectore_search import retrieve_anime_recommendations
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
from dotenv import load_dotenv
load_dotenv()

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
            content=f"""Please refine the following  input into a detailed and precise query:

    Input: {state['input_text']}"""
        )
    ]
    
    llm_model = state.get('llm_model', 'Groq')
    redefine_input_llm, _ = get_llm_chains(llm_model)

    response = redefine_input_llm.invoke(messages)
    state['redefine_input_content'] = response.refined_query
    return state

def anime_semantic_search(state: GraphState) -> GraphState:
    """
    Performs semantic search to retrieve relevant anime recommendations from the vector database.
    """
    query = state['redefine_input_content']
    vector_source = state.get('vector_source', 'Pinecone')
    # Always use HuggingFace embeddings
    context = retrieve_anime_recommendations(query=query, k=8, source=vector_source)
    state['context'] = context
    return state
    


async def get_tavily_search(anime: str) -> str:
    """
    Performs web search to retrieve relevant details about recommended anime.
    """
    web_search_tool = TavilySearch(max_result=3,include_answer="advanced",
    include_images=True)
    tavily_results = await web_search_tool.ainvoke(
            {
                "query":f" Give me details about {anime} anime with its description, total number of episodes, rating, genre, demographic"
            })
    return tavily_results   
    


def anime_recommendation(state: GraphState) -> GraphState:
    """
    Retrieves details about the recommended anime from the web search results.
    """
    recommended_anime = [anime.metadata['title'] for anime in state['context']]
    async def search_all():
        results = [get_tavily_search(anime) for anime in recommended_anime]
        return await asyncio.gather(*results)
    tavily_results = asyncio.run(search_all())
    """
    Generates final anime recommendations based on the refined query and retrieved context.
    """
    query = state['redefine_input_content']
    context = state['context']
    messages = [
    SystemMessage(
            content="""
                    You are an expert anime recommendation specialist with deep knowledge of anime across all genres, demographics, and eras.

                    You will be given TWO information sources:
                    1. Retrieved Context (from internal database / vector store)
                    2. Web Search Results (external, real-time information)

                    SOURCE PRIORITY RULES (VERY IMPORTANT):
                    - The Retrieved Context is the PRIMARY and authoritative source
                    - Web Search Results are SECONDARY and should ONLY be used to:
                    • Verify missing fields (score, episodes, rating, image URL)
                    • Resolve outdated or conflicting information
                    • Fill gaps ONLY if the anime already exists in Retrieved Context
                    - DO NOT introduce any new anime titles from Web Search Results alone
                    - NEVER recommend anime not present in the Retrieved Context

                    TASK:
                    Analyze the user's query and recommend the TOP 5 most relevant anime titles.

                    SELECTION RULES:
                    - Recommend ONLY anime found in the Retrieved Context
                    - Rank based on relevance to the user’s preferences
                    - Prioritize quality, thematic alignment, and genre match
                    - Avoid duplicates or loosely related titles

                    DATA EXTRACTION RULES:
                    - Prefer Retrieved Context values first
                    - If a field is missing or unclear, use Web Search Results ONLY for that anime
                    - If conflicts exist, choose the most recent and reliable information

                    REQUIRED DETAILS PER ANIME:
                    1. Title (official name)
                    2. Description (complete synopsis)
                    3. Score (rating score out of 10)
                    4. Image URL (poster image)
                    5. Episodes (total episode count)
                    6. Rating (age classification)
                    7. Genres (all applicable genres)
                    8. Demographic (Shounen, Seinen, Shoujo, etc.)

                    OUTPUT FORMAT:
                    - Return exactly 5 anime recommendations
                    - Use clear, structured formatting
                    - Do not include explanations or meta commentary
                    """
                    ),
            HumanMessage(
                content=f"""

            User's  Query:
            {query}

            Retrieved Context:
            {context}

            Web Search Results:
            {tavily_results}

            Please extract and return the 5 best matching anime with all required details."""
            )
                ]
    llm_model = state.get('llm_model', 'Groq')
    _, recommended_Anime_llm = get_llm_chains(llm_model)

    # The LLM is already bound with the RecommendedAnime schema which contains the list of AnimeDetails
    state['recommended_anime'] = recommended_Anime_llm.invoke(messages).anime_titles
    return state




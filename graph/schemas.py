from typing import List, Optional
from pydantic import BaseModel, Field


class RefinedQuery(BaseModel):
    '''Schema to refine the user's input text into a precise search query'''
    refined_query: str = Field(description='The refined and contextualized search query based on user input')


class RecommendedAnime(BaseModel):
    """Recommended anime information based on context"""
    anime_titles: Optional[List[str]] = Field(description='List of recommended anime titles')
    #IMDB_Rating: Optional[int] = Field(description='IMDB rating for the anime')

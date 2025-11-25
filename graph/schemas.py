from typing import List, Optional
from pydantic import BaseModel, Field


class RefinedQuery(BaseModel):
    '''Schema to refine the user's input text into a precise search query'''
    refined_query: str = Field(description='The refined and contextualized search query based on user input')


class AnimeDetails(BaseModel):
    """Detailed information about a recommended anime"""
    title: str = Field(description="The title of the anime")
    description: str = Field(description="A brief synopsis of the anime")
    score: Optional[float] = Field(description="The score of the anime")
    image_url: Optional[str] = Field(description="The URL of the anime's image")
    episodes: Optional[float] = Field(description="The number of episodes")
    rating: Optional[str] = Field(description="The age rating of the anime")
    genres: Optional[str] = Field(description="The genres of the anime")
    demographic: Optional[str] = Field(description="The demographic of the anime")

class RecommendedAnime(BaseModel):
    """Recommended anime information based on context"""
    anime_titles: List[AnimeDetails] = Field(description='List of recommended anime with details')

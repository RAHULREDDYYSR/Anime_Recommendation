"""
API utilities for fetching anime data from external sources.
"""
import requests
import streamlit as st


@st.cache_data(ttl=3600)
def get_anime_image(anime_name: str):
    """
    Fetch anime information and image from Jikan API (MyAnimeList).
    
    Args:
        anime_name: Name of the anime to search for
        
    Returns:
        dict: Anime information including image URL, score, episodes, etc.
        None: If anime not found or API error
    """
    try:
        # Search for anime by name
        search_url = f"https://api.jikan.moe/v4/anime?q={anime_name}&limit=1"
        response = requests.get(search_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                anime_data = data['data'][0]
                synopsis = anime_data.get('synopsis', '')
                
                return {
                    'image_url': anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                    'title': anime_data.get('title'),
                    'score': anime_data.get('score'),
                    'episodes': anime_data.get('episodes'),
                    'year': anime_data.get('year'),
                    'synopsis': anime_data.get('synopsis', '')  # Full synopsis without truncation
                }
    except Exception as e:
        print(f"Error fetching image for {anime_name}: {e}")
    
    return None

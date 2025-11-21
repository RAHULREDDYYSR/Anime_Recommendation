"""
UI components for the Streamlit anime recommendation app.
"""
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.api_utils import get_anime_image


def render_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        .anime-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """
    Render the sidebar with app information and settings.
    
    Returns:
        tuple: (num_recommendations, show_images)
    """
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system uses:
        - **LangGraph** for workflow
        - **LLMs** for query refinement
        - **Semantic Search** for recommendations
        - **Pinecone** vector database
        """)
        
        st.header("Settings")
        num_recommendations = st.slider("Number of recommendations", 1, 10, 10)
        show_images = st.checkbox("Show anime images", value=True)
        
        return num_recommendations, show_images


def fetch_anime_info_batch(anime_names: list):
    """
    Fetch anime information for multiple anime in parallel.
    
    Args:
        anime_names: List of anime names to fetch
        
    Returns:
        dict: Dictionary mapping anime names to their info
    """
    anime_info_dict = {}
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_anime = {
            executor.submit(get_anime_image, anime): anime 
            for anime in anime_names
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_anime):
            anime = future_to_anime[future]
            try:
                anime_info_dict[anime] = future.result()
            except Exception as e:
                print(f"Error fetching {anime}: {e}")
                anime_info_dict[anime] = None
    
    return anime_info_dict


def render_anime_card_with_image(idx: int, anime_name: str, anime_info: dict = None):
    """
    Render an anime recommendation card with image and metadata.
    
    Args:
        idx: Index number for the recommendation
        anime_name: Name of the anime
        anime_info: Pre-fetched anime information (optional)
    """
    import urllib.parse
    
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display anime image
            if anime_info and anime_info.get('image_url'):
                st.image(anime_info['image_url'], width=150)
            else:
                st.image("https://via.placeholder.com/150x200?text=No+Image", width=150)
        
        with col2:
            # Create Google search link
            search_query = urllib.parse.quote(f"{anime_name} anime")
            google_search_url = f"https://www.google.com/search?q={search_query}"
            
            # Display title with Google search link
            st.markdown(f"### {idx}. [{anime_name}]({google_search_url})")
            
            if anime_info:
                if anime_info.get('score'):
                    st.markdown(f"⭐ **Score:** {anime_info['score']}/10")
                if anime_info.get('episodes'):
                    st.markdown(f"📺 **Episodes:** {anime_info['episodes']}")
                if anime_info.get('year'):
                    st.markdown(f"📅 **Year:** {anime_info['year']}")
                if anime_info.get('synopsis'):
                    with st.expander("📖 Synopsis"):
                        st.write(anime_info['synopsis'])
        
        st.markdown("---")


def render_anime_card_simple(idx: int, anime_name: str):
    """
    Render a simple anime recommendation card without image.
    
    Args:
        idx: Index number for the recommendation
        anime_name: Name of the anime
    """
    st.markdown(f"**{idx}. {anime_name}**")


def render_recommendations(recommendations: list, num_recommendations: int, show_images: bool, layout_style: str = "List"):
    """
    Render the list of anime recommendations.
    
    Args:
        recommendations: List of anime names
        num_recommendations: Number of recommendations to display
        show_images: Whether to show images or not
        layout_style: Layout style (for future expansion)
    """
    st.subheader("🎬 Recommended Anime:")
    
    if recommendations:
        if show_images:
            # Fetch all anime info in parallel
            anime_list = recommendations[:num_recommendations]
            with st.spinner("Loading anime details..."):
                anime_info_dict = fetch_anime_info_batch(anime_list)
            
            # Render cards with pre-fetched data
            for idx, anime in enumerate(anime_list, 1):
                render_anime_card_with_image(idx, anime, anime_info_dict.get(anime))
        else:
            for idx, anime in enumerate(recommendations[:num_recommendations], 1):
                render_anime_card_simple(idx, anime)
    else:
        st.warning("No recommendations found. Try a different query!")


def render_footer():
    """Render the app footer."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Built with ❤️ using LangChain")
    with col2:
        st.markdown("Powered by Groq & Pinecone")
    with col3:
        st.markdown("[GitHub](https://github.com/RAHULREDDYYSR/Anime_Recommendation)")

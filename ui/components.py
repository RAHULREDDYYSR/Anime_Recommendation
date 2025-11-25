"""
UI components for the Streamlit anime recommendation app.
"""
import streamlit as st
import urllib.parse

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


def render_anime_card_with_image(idx: int, anime: object):
    """
    Render an anime recommendation card with image and metadata.
    
    Args:
        idx: Index number for the recommendation
        anime: AnimeDetails object containing anime information
    """
    # Handle both Pydantic object and dict
    if isinstance(anime, dict):
        title = anime.get('title', 'Unknown Title')
        image_url = anime.get('image_url')
        score = anime.get('score')
        episodes = anime.get('episodes')
        rating = anime.get('rating')
        genres = anime.get('genres')
        description = anime.get('description')
    else:
        title = getattr(anime, 'title', 'Unknown Title')
        image_url = getattr(anime, 'image_url', None)
        score = getattr(anime, 'score', None)
        episodes = getattr(anime, 'episodes', None)
        rating = getattr(anime, 'rating', None)
        genres = getattr(anime, 'genres', None)
        description = getattr(anime, 'description', None)

    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display anime image
            if image_url:
                st.image(image_url, width=150)
            else:
                st.image("https://via.placeholder.com/150x200?text=No+Image", width=150)
        
        with col2:
            # Create Google search link
            search_query = urllib.parse.quote(f"{title} anime")
            google_search_url = f"https://www.google.com/search?q={search_query}"
            
            # Display title with Google search link
            st.markdown(f"### {idx}. [{title}]({google_search_url})")
            
            # Display metadata
            meta_cols = st.columns(3)
            with meta_cols[0]:
                if score:
                    st.markdown(f"⭐ **Score:** {score}/10")
            with meta_cols[1]:
                if episodes:
                    st.markdown(f"📺 **Episodes:** {episodes}")
            with meta_cols[2]:
                if rating:
                    st.markdown(f"🔞 **Rating:** {rating}")
            
            if genres:
                st.markdown(f"🎭 **Genres:** {genres}")

            if description:
                with st.expander("📖 Synopsis"):
                    st.write(description)
        
        st.markdown("---")


def render_anime_card_simple(idx: int, anime: object):
    """
    Render a simple anime recommendation card without image.
    
    Args:
        idx: Index number for the recommendation
        anime: AnimeDetails object or dict
    """
    if isinstance(anime, dict):
        title = anime.get('title', 'Unknown Title')
    else:
        title = getattr(anime, 'title', 'Unknown Title')
        
    st.markdown(f"**{idx}. {title}**")


def render_recommendations(recommendations: list, num_recommendations: int, show_images: bool, layout_style: str = "List"):
    """
    Render the list of anime recommendations.
    
    Args:
        recommendations: List of AnimeDetails objects
        num_recommendations: Number of recommendations to display
        show_images: Whether to show images or not
        layout_style: Layout style (for future expansion)
    """
    st.subheader("🎬 Recommended Anime:")
    
    if recommendations:
        # Limit recommendations
        anime_list = recommendations[:num_recommendations]
        
        for idx, anime in enumerate(anime_list, 1):
            if show_images:
                render_anime_card_with_image(idx, anime)
            else:
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

"""
Main Streamlit application for anime recommendations.
"""
import streamlit as st
import time
from graph.graph import app
from ui.components import (
    render_custom_css,
    render_sidebar,
    render_recommendations,
    render_footer
)

# Page configuration
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide"
)

# Apply custom styling
render_custom_css()

# Header
st.title("🎌 Anime Recommendation System")
st.markdown("Powered by AI and semantic search")

# Sidebar with settings
vector_source, llm_model = render_sidebar()

# Main interface
user_query = st.text_input(
    "What kind of anime are you looking for?",
    placeholder="e.g., I want a shonen anime with good fights"
)

if st.button("Get Recommendations", type="primary") or user_query:
    if user_query:
        with st.spinner("Finding the perfect anime for you..."):
            start_time = time.time()
            
            try:
                # Get recommendations from the graph
                result = app.invoke({"input_text": user_query, "vector_source": vector_source, "llm_model": llm_model})
                end_time = time.time()
                
                # Display success message
                st.success(f"✨ Found recommendations in {end_time - start_time:.2f} seconds!")
                
                # Render recommendations with images
                recommendations = result.get('recommended_anime', [])
                render_recommendations(recommendations)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to get recommendations!")

# Footer
render_footer()
